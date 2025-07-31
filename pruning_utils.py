import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import math

import torch_pruning as tp

def pruning_model(model, px, conv1=False):
    # print('start unstructured pruning for all conv layers')
    parameters_to_prune = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                parameters_to_prune.append((m, 'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def check_sparsity(model, conv1=True):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    sum_list = sum_list + float(m.weight.nelement())
                    zero_sum = zero_sum + float(torch.sum(m.weight == 0))
                else:
                    print('skip conv1 for sparsity checking')
            else:
                sum_list = sum_list + float(m.weight.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    print('* remain weight = ', 100 * (1 - zero_sum / sum_list), '%')

    return 100 * (1 - zero_sum / sum_list)

def prune_model_custom_fillback(model, mask_dict, conv1=False, criteria="remain", train_loader=None, init_weight=None,
                                trained_weight=None, return_mask_only=False, strict=True, fillback_rate=0.0):
    feature_maps = []
    try:
        model.load_state_dict(trained_weight, strict=strict)
    except:
        for key in list(trained_weight.keys()):
            if ('mask' in key):
                trained_weight[key[:-5]] = trained_weight[key[:-5] + "_orig"] * trained_weight[key]
                del trained_weight[key[:-5] + "_orig"]
                del trained_weight[key]
        model.load_state_dict(trained_weight, strict=strict)

    def hook(module, input, output):
        feature_maps.append(output)

    image, label = next(iter(train_loader))
    handles = []
    masks = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                handles.append(m.register_forward_hook(hook))
                device = m.weight.data.device
    output = model(image.to(device))
    loss = torch.nn.CrossEntropyLoss()(output, label.to(output.device))
    counter = 0

    for i, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                mask = mask_dict[name + '.weight_mask']
                mask = mask.view(mask.shape[0], -1)
                count = torch.sum(mask, 1)  # [C]
                # sparsity = torch.sum(mask) / mask.numel()
                num_channel = (count.sum().float() / mask.shape[1]).item()
                # print(num_channel)
                # print(mask.shape[0])
                # print(fillback_rate)
                # print(mask.shape[0] - num_channel)
                # print((mask.shape[0] - num_channel) * fillback_rate)
                int_channel = int(num_channel + (mask.shape[0] - num_channel) * fillback_rate)
                frac_channel = num_channel - int_channel
                # print(mask.shape)
                # print(int_channel)
                if criteria == 'remain':
                    # print(mask.shape[0] - int_channel)
                    threshold, _ = torch.kthvalue(count, max(mask.shape[0] - int_channel, 1))

                    mask[torch.where(count > threshold)[0]] = 1
                    mask[torch.where(count < threshold)[0]] = 0
                    tensor = torch.where(count == threshold)[0]
                    perm = torch.randperm(tensor.size(0))
                    idx = perm[0]
                    samples = tensor[idx]
                    mask[samples] = 1

                elif criteria == 'magnitude':
                    mask = mask_dict[name + '.weight_mask']
                    count = trained_weight[name + '.weight'].view(mask.shape[0], -1).abs().sum(1)
                    if (mask.shape[0] - int_channel) > 0:
                        threshold, _ = torch.kthvalue(count, mask.shape[0] - int_channel)
                        mask[torch.where(count > threshold)[0]] = 1
                        mask[torch.where(count < threshold)[0]] = 0
                        tensor = torch.where(count == threshold)[0]
                        perm = torch.randperm(tensor.size(0))
                        idx = perm[0]
                        samples = tensor[idx]
                        mask[samples] = 1
                    else:
                        mask[:, :] = 1

                elif criteria == 'l1':
                    mask = mask_dict[name + '.weight_mask']
                    count = feature_maps[counter].view(mask.shape[0], -1).abs().sum(1)
                    threshold, _ = torch.kthvalue(count, mask.shape[0] - int_channel)

                    mask[torch.where(count > threshold)[0]] = 1
                    mask[torch.where(count < threshold)[0]] = 0
                    tensor = torch.where(count == threshold)[0]
                    perm = torch.randperm(tensor.size(0))
                    idx = perm[0]
                    samples = tensor[idx]
                    mask[samples] = 1

                elif criteria == 'l2':
                    mask = mask_dict[name + '.weight_mask']
                    count = (feature_maps[counter].view(mask.shape[0], -1).abs() ** 2).sum(1)
                    threshold, _ = torch.kthvalue(count, mask.shape[0] - int_channel)

                    mask[torch.where(count > threshold)[0]] = 1
                    mask[torch.where(count < threshold)[0]] = 0
                    tensor = torch.where(count == threshold)[0]
                    perm = torch.randperm(tensor.size(0))
                    idx = perm[0]
                    samples = tensor[idx]
                    mask[samples] = 1
                elif criteria == 'saliency':
                    mask = mask_dict[name + '.weight_mask']
                    count = (feature_maps[counter] *
                             torch.autograd.grad(loss, feature_maps[counter], retain_graph=True, only_inputs=True)[
                                 0]).view(mask.shape[0], -1).abs().sum(1)
                    threshold, _ = torch.kthvalue(count, mask.shape[0] - int_channel)

                    mask[torch.where(count > threshold)[0]] = 1
                    mask[torch.where(count < threshold)[0]] = 0
                    tensor = torch.where(count == threshold)[0]
                    perm = torch.randperm(tensor.size(0))
                    idx = perm[0]
                    samples = tensor[idx]
                    mask[samples] = 1

                # Modify to cater for "prune_model_custom"
                # Former Version
                # if not return_mask_only:
                #    m.weight.data = init_weight[name + ".weight"]
                #    mask = mask.view(*mask_dict[name+'.weight_mask'].shape)
                #    print('pruning layer with custom mask:', name)
                #    prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))
                # else:
                #    masks[name] = mask

                # New Version
                mask = mask.view(*mask_dict[name + '.weight_mask'].shape)
                if not return_mask_only:
                    m.weight.data = init_weight[name + ".weight"]
                    print('pruning layer with custom mask:', name)
                    prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))
                else:
                    masks[name + '.weight_mask'] = mask

    for h in handles:
        h.remove()

    if return_mask_only:
        return masks

def prune_model_custom(model, mask_dict, conv1=False):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                # print('pruning layer with custom mask:', name)
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name + '.weight_mask'].to(m.weight.device))

class FedLTHPruner:
    def __init__(self, start_ratio, end_ratio, channel_sparsity,device, speed=0.2, min_inscrease=0.01):
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.speed = speed
        self.min_inscrease = min_inscrease
        self.ratio = start_ratio
        self.channel_sparsity = channel_sparsity

        self.unpruned_flag=False
        self.device=device

    def _update_ratio(self):
        delta = self.end_ratio - self.ratio
        change = delta * (1 - math.exp(-self.speed))
        if change < self.min_inscrease:
            change=self.min_inscrease
        self.ratio += change

    def _extract_mask(self,model_dict):
        new_dict = {}
        for key in model_dict.keys():
            if 'mask' in key:
                new_dict[key] = model_dict[key]
        return new_dict

    def unstuctured_prune(self, model, conv1=False):
        self._update_ratio()
        self.unpruned_flag=True
        pruning_model(model, self.ratio, conv1=conv1)
    
    def structured_prune(self, model, weight_with_mask, trace_data_loader, criterion, num_classes):
        prune_mask = self._refill(model,None,weight_with_mask,trace_data_loader, mask_only=True) # init_weight can be None when return mask_only
        prune_model_custom(model, prune_mask, conv1=False)
        self.remove_prune(model, conv1=False)
        model.zero_grad()
        trace_data=next(iter(trace_data_loader))
        model, sparsity = self._tp_prune(model,trace_data,criterion,num_classes,self.channel_sparsity, imp_strategy='Magnitude',degree=1)
        return model, sparsity

    def _refill(self,model,init_weight,weight_with_mask,train_loader,mask_only=False):
        current_mask = self._extract_mask(weight_with_mask)
        if mask_only:
            return prune_model_custom_fillback(model, current_mask, criteria='remain', train_loader=train_loader,trained_weight=model.state_dict(),init_weight=init_weight,return_mask_only=True)
        else:
            return prune_model_custom_fillback(model, current_mask, criteria='remain', train_loader=train_loader,trained_weight=model.state_dict(),init_weight=init_weight)
    
    # Torch_pruning结构化剪枝,imp_strategy=Magnitude/Taylor(Default)/Hessian
    def _tp_prune(self, model, trace_data,criterion,num_classes,
                    ratio, imp_strategy='Taylor', degree=2, iterative_steps=1, show_step=False, show_group=False):
        # print(f'Channel Ratio:{ratio}')
        if imp_strategy == 'Magnitude':
            assert degree == 1 or degree == 2  # degree must be 1 or 2
            imp = tp.importance.MagnitudeImportance(p=degree)
        elif imp_strategy == 'Taylor':
            imp = tp.importance.TaylorImportance()
        elif imp_strategy == 'Hessian':
            imp = tp.importance.HessianImportance()
        else:
            return

        # Ignore some layers, e.g., the output layer
        ignored_layers = []
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
                ignored_layers.append(m)  # DO NOT prune the final classifier!

        trace_input,trace_label=trace_data
        trace_input=trace_input.to(self.device)
        # Initialize a pruner
        pruner = tp.pruner.MagnitudePruner(
                model,
                trace_input,
                importance=imp,
                iterative_steps=iterative_steps,
                pruning_ratio=ratio,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
                ignored_layers=ignored_layers,
                )

        # prune the model, iteratively if necessary.
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, trace_input)
        macs, nparams = base_macs, base_nparams
        for i in range(iterative_steps):
            if isinstance(imp, tp.importance.TaylorImportance):
                # # A dummy loss, please replace it with your loss function and data!
                # loss = model(trace_input).sum()
                # loss.backward()  # before pruner.step()
                output = model(trace_input)
                loss = criterion(output, trace_label)
                loss.backward()  # before pruner.step()
            if show_group:
                for group in pruner.step(interactive=True):  # Warning: groups must be handled sequentially. Do not keep them as a list.
                    # print(group)
                    group.prune()
            else:
                pruner.step()
            macs, nparams = tp.utils.count_ops_and_params(model, trace_input)
            if show_step:
                print('Current sparsity:' + str(100 * nparams / base_nparams) + '%')

        return model, nparams/base_nparams
    
    def remove_prune(self, model, conv1=False):
    # print('remove pruning')
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                if (name == 'conv1' and conv1) or (name != 'conv1'):
                    prune.remove(m, 'weight')