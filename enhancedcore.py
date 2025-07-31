from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.utils.dataset.functional import split_indices
from fedlab.utils.functional import partition_report
import torch
from torch.utils.data import SubsetRandomSampler

from fedlab.utils import Logger, SerializationTool

import pickle
from gzip import compress,decompress

class ModifiedSubsetSerialTrainer(SerialClientTrainer):
    """
    Modified Details: upload model rather than model params
    """

    """Train multiple clients in a single process.

    Customize :meth:`_get_dataloader` or :meth:`_train_alone` for specific algorithm design in clients.

    Args:
        model (torch.nn.Module): Model used in this federation.
        dataset (torch.utils.data.Dataset): Local dataset for this group of clients.
        data_slices (list[list]): Subset of indices of dataset.
        logger (Logger, optional): Object of :class:`Logger`.
        cuda (bool): Use GPUs or not. Default: ``False``.
        args (dict, optional): Uncertain variables.

    .. note::
        ``len(data_slices) == client_num``, that is, each sub-index of :attr:`dataset` corresponds to a client's local dataset one-by-one.
    """

    def __init__(self,
                 model,
                 dataset,
                 data_slices,
                 logger=None,
                 cuda=False,
                 gpu=None,
                 args=None) -> None:

        super(ModifiedSubsetSerialTrainer, self).__init__(model=model,
                                                  num_clients=len(data_slices),
                                                  cuda=cuda)
        self.gpu = gpu

        self.dataset = dataset
        self.data_slices = data_slices  # [0, client_num)
        self.args = args
        self.logger = logger if logger is not None else Logger()
        self._LOGGER = self.logger

    def local_process(self, id_list, model_data):
        """Train local model with different dataset according to client id in ``id_list``.

        Args:
            id_list (list[int]): Client id in this training serial.
            model_data (compress(pick.dumps(model))): Serialized model.
        """
        self.param_list = []
        #rebuild_model=pickle.loads(decompress(model_data))
        #self._model=rebuild_model
        #model_parameters = SerializationTool.serialize_model(rebuild_model)
        model_parameters=model_data[0]

        self._LOGGER.info(
            "Local training with client id list: {}".format(id_list))
        for idx in id_list:
            self._LOGGER.info(
                "Starting training procedure of client [{}]".format(idx))

            data_loader = self._get_dataloader(client_id=idx)
            self._train_alone(model_parameters=model_parameters,
                              train_loader=data_loader)
            self.param_list.append(self.model_parameters)
        return self.param_list

    def _get_dataloader(self, client_id):
        """Return a training dataloader used in :meth:`train` for client with :attr:`id`

        Args:
            client_id (int): :attr:`client_id` of client to generate dataloader

        Note:
            :attr:`client_id` here is not equal to ``client_id`` in global FL setting. It is the index of client in current :class:`SerialTrainer`.

        Returns:
            :class:`DataLoader` for specific client's sub-dataset
        """
        batch_size = self.args["batch_size"]

        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(indices=self.data_slices[client_id]),
            batch_size=batch_size,
            num_workers=8)

        return train_loader

    def _train_alone(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        epochs, lr = self.args["epochs"], self.args["lr"]
        SerializationTool.deserialize_model(self._model, model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._model.train()

        for _ in range(epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                output = self.model(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.set_model(SerializationTool.serialize_model(self._model))
        return self.model_parameters