import os
import sys

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

##########################################################################3
def pruning_bert2(model, px, model_type='wav2vec_small'):
    """
    prune out wav2vec 2.0 BERT: 12 transformer layers for BASE, and 24 
                                transformer layers for LARGE

    note: position encoding and projection heads are not pruned. 
    """

    if model_type == 'wav2vec_small':
        num_transformer_blocks = 12
    elif model_type == 'libri960_big':
        num_transformer_blocks = 24
    elif model_type == 'xlsr_53_56k':
        num_transformer_blocks = 24
    else:
        print('model type {} not supported'.format(model_type))
        exit()
    print('num_transformer_blocks is', num_transformer_blocks)

    parameters_to_prune =[]
    # position encoding for BERT
    #model.encoder.pos_conv[0].weight = nn.Parameter(model.encoder.pos_conv[0].weight,
    #                                                requires_grad=True) # hack to make it work
    #model.encoder.pos_conv[0].bias = nn.Parameter(model.encoder.pos_conv[0].bias,
    #                                                requires_grad=True) # hack
    #parameters_to_prune.append((model.encoder.pos_conv[0], 'weight'))
    #parameters_to_prune.append((model.encoder.pos_conv[0], 'bias'))

    for ii in range(num_transformer_blocks):
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.k_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.k_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.v_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.v_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.q_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.q_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.out_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].self_attn.out_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].fc1, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].fc1, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].fc2, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].fc2, 'bias'))

    parameters_to_prune = tuple(parameters_to_prune)

    import contextualized_prune
    #prune.global_unstructured(
    #    parameters_to_prune,
    #    pruning_method=prune.L1Unstructured,
    #    amount=px,
    #)
    contextualized_prune.global_unstructured(
        parameters_to_prune,
        pruning_method=contextualized_prune.LGUnstructured,
        amount=px,
    )

##########################################################################3

def add_pruning_argument(parser):
    parser.add_argument('--prune_rate', default=0.2, type=float)

    return parser

def unprune_transformer_tts(model):
    """
    remove pruning forward pre-hook. This is useful when we want to tweek the learned pruned mask.
    """
    parameters_to_prune =[]

    # encoder
    for ii in range(6):
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_q))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_k))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_v))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_out))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].feed_forward.w_1))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].feed_forward.w_2))

    # prenet 
    parameters_to_prune.append((model.tts.decoder.embed[0][0].prenet[0][0]))
    parameters_to_prune.append((model.tts.decoder.embed[0][0].prenet[1][0]))
    parameters_to_prune.append((model.tts.decoder.embed[0][1]))

    # decoder 
    for ii in range(6):
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_q))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_k))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_v))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_out))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_q))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_k))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_v))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_out))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].feed_forward.w_1))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].feed_forward.w_2))

    # postnet 
    for ii in range(5):
        parameters_to_prune.append((model.tts.postnet.postnet[ii][0]))


    for ii in range(0, len(parameters_to_prune)-5): # applying both weight+bias masks
        prune.remove(parameters_to_prune[ii], 'weight')
        prune.remove(parameters_to_prune[ii], 'bias')
    for ii in range(len(parameters_to_prune)-5, len(parameters_to_prune)):
        prune.remove(parameters_to_prune[ii], 'weight')

def pruning_transformer_tts(model, px):
    """
    prune out transformer-TTS: encoder, prenet, decoder, postnet

    note: position encoding and projection heads are not pruned. 
    """

    parameters_to_prune =[]

    # encoder 
    for ii in range(6):
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_q, 'weight'))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_q, 'bias'))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_k, 'weight'))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_k, 'bias'))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_v, 'weight'))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_v, 'bias'))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_out, 'weight'))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].self_attn.linear_out, 'bias'))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].feed_forward.w_1, 'weight'))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].feed_forward.w_1, 'bias'))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].feed_forward.w_2, 'weight'))
        parameters_to_prune.append((model.tts.encoder.encoders[ii].feed_forward.w_2, 'bias'))

    # prenet 
    parameters_to_prune.append((model.tts.decoder.embed[0][0].prenet[0][0], 'weight'))
    parameters_to_prune.append((model.tts.decoder.embed[0][0].prenet[0][0], 'bias'))
    parameters_to_prune.append((model.tts.decoder.embed[0][0].prenet[1][0], 'weight'))
    parameters_to_prune.append((model.tts.decoder.embed[0][0].prenet[1][0], 'bias'))
    parameters_to_prune.append((model.tts.decoder.embed[0][1], 'weight'))
    parameters_to_prune.append((model.tts.decoder.embed[0][1], 'bias'))

    # decoder 
    for ii in range(6):
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_q, 'weight'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_q, 'bias'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_k, 'weight'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_k, 'bias'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_v, 'weight'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_v, 'bias'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_out, 'weight'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].self_attn.linear_out, 'bias'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_q, 'weight'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_q, 'bias'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_k, 'weight'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_k, 'bias'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_v, 'weight'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_v, 'bias'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_out, 'weight'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].src_attn.linear_out, 'bias'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].feed_forward.w_1, 'weight'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].feed_forward.w_1, 'bias'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].feed_forward.w_2, 'weight'))
        parameters_to_prune.append((model.tts.decoder.decoders[ii].feed_forward.w_2, 'bias'))

    # postnet 
    for ii in range(5):
        parameters_to_prune.append((model.tts.postnet.postnet[ii][0], 'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def see_weight_rate(model):

    # encoder
    sum_list_1, zero_sum_1 = 0, 0
    for ii in range(6):
        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].self_attn.linear_q.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].self_attn.linear_q.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].self_attn.linear_q.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].self_attn.linear_q.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].self_attn.linear_k.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].self_attn.linear_k.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].self_attn.linear_k.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].self_attn.linear_k.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].self_attn.linear_v.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].self_attn.linear_v.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].self_attn.linear_v.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].self_attn.linear_v.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].self_attn.linear_out.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].self_attn.linear_out.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].self_attn.linear_out.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].self_attn.linear_out.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].feed_forward.w_1.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].feed_forward.w_1.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].feed_forward.w_1.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].feed_forward.w_1.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].feed_forward.w_2.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].feed_forward.w_2.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.encoder.encoders[ii].feed_forward.w_2.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.encoder.encoders[ii].feed_forward.w_2.bias == 0))

    # prenet
    sum_list_1 = sum_list_1 + float(model.tts.decoder.embed[0][0].prenet[0][0].weight.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.embed[0][0].prenet[0][0].weight == 0))
    sum_list_1 = sum_list_1 + float(model.tts.decoder.embed[0][0].prenet[0][0].bias.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.embed[0][0].prenet[0][0].bias == 0))

    sum_list_1 = sum_list_1 + float(model.tts.decoder.embed[0][0].prenet[1][0].weight.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.embed[0][0].prenet[1][0].weight == 0))
    sum_list_1 = sum_list_1 + float(model.tts.decoder.embed[0][0].prenet[1][0].bias.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.embed[0][0].prenet[1][0].bias == 0))

    sum_list_1 = sum_list_1 + float(model.tts.decoder.embed[0][1].weight.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.embed[0][1].weight == 0))
    sum_list_1 = sum_list_1 + float(model.tts.decoder.embed[0][1].bias.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.embed[0][1].bias == 0))

    # decoder
    for ii in range(6):
        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].self_attn.linear_q.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].self_attn.linear_q.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].self_attn.linear_q.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].self_attn.linear_q.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].self_attn.linear_k.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].self_attn.linear_k.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].self_attn.linear_k.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].self_attn.linear_k.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].self_attn.linear_v.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].self_attn.linear_v.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].self_attn.linear_v.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].self_attn.linear_v.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].self_attn.linear_out.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].self_attn.linear_out.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].self_attn.linear_out.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].self_attn.linear_out.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].src_attn.linear_q.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].src_attn.linear_q.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].src_attn.linear_q.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].src_attn.linear_q.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].src_attn.linear_k.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].src_attn.linear_k.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].src_attn.linear_k.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].src_attn.linear_k.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].src_attn.linear_v.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].src_attn.linear_v.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].src_attn.linear_v.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].src_attn.linear_v.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].src_attn.linear_out.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].src_attn.linear_out.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].src_attn.linear_out.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].src_attn.linear_out.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].feed_forward.w_1.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].feed_forward.w_1.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].feed_forward.w_1.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].feed_forward.w_1.bias == 0))

        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].feed_forward.w_2.weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].feed_forward.w_2.weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.decoder.decoders[ii].feed_forward.w_2.bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.decoder.decoders[ii].feed_forward.w_2.bias == 0))

    # postnet 
    for ii in range(5):
        sum_list_1 = sum_list_1 + float(model.tts.postnet.postnet[ii][0].weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.postnet.postnet[ii][0].weight == 0))

    total_zero_rate = 100 * zero_sum_1 / sum_list_1
    #print('sum_list_1 is %d' % sum_list_1)
    print('zero rate is %.2f' % total_zero_rate)

    return total_zero_rate


def apply_pruning_mask(model, mask_dict, prune_component='bert', model_type='wav2vec_small', fp16=False):
    """
    apply pruning mask to wav2vec 2.0. We only prune either just the quantizer or just the BERT.
    """

    if prune_component not in ['bert', 'quantizer']:
        print('{} not supported'.format(prune_component))
        exit()

    if model_type == 'wav2vec_small':
        num_transformer_blocks = 12
    elif model_type == 'libri960_big' or model_type == 'xlsr_53_56k':
        num_transformer_blocks = 24
    else:
        print('model type {} not supported'.format(model_type))
        exit()

    #if fp16:
    #    model = model.half() # do this for fp16

    parameters_to_prune =[]
    mask_list_w, mask_list_b = [], [] # maks list for weight and bias

    if prune_component == 'feature_extractor':
        # feature_extractor
        for ii in range(7):
            parameters_to_prune.append(model.feature_extractor.conv_layers[ii][0])
            mask_list_w.append(mask_dict['feature_extractor.conv_layers.' + str(ii) + '.0.weight_mask'])

        parameters_to_prune.append(model.post_extract_proj)
        mask_list_w.append(mask_dict['post_extract_proj.weight_mask'])
        mask_list_b.append(mask_dict['post_extract_proj.bias_mask'])

    if prune_component == 'quantizer':
        parameters_to_prune.append(model.quantizer.weight_proj)
        mask_list_w.append(mask_dict['quantizer.weight_proj.weight_mask'])
        mask_list_w.append(mask_dict['quantizer.weight_proj.bias_mask'])

        parameters_to_prune.append(model.project_q)
        mask_list_w.append(mask_dict['project_q.weight_mask'])
        mask_list_w.append(mask_dict['project_q.bias_mask'])

    # BERT
    #model.encoder.pos_conv[0].weight = nn.Parameter(model.encoder.pos_conv[0].weight,
    #                                                requires_grad=True) # hack to make it work
    #model.encoder.pos_conv[0].bias = nn.Parameter(model.encoder.pos_conv[0].bias,
    #                                                requires_grad=True) # hack
    #parameters_to_prune.append(model.encoder.pos_conv[0])
    #mask_list_w.append(mask_dict['encoder.pos_conv.0.weight_mask'])
    #mask_list_b.append(mask_dict['encoder.pos_conv.0.bias_mask'])

    if prune_component == 'bert':
        for ii in range(num_transformer_blocks):
            parameters_to_prune.append(model.encoder.layers[ii].self_attn.k_proj)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.k_proj.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.k_proj.bias_mask'])
            parameters_to_prune.append(model.encoder.layers[ii].self_attn.v_proj)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.v_proj.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.v_proj.bias_mask'])
            parameters_to_prune.append(model.encoder.layers[ii].self_attn.q_proj)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.q_proj.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.q_proj.bias_mask'])
            parameters_to_prune.append(model.encoder.layers[ii].self_attn.out_proj)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.out_proj.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.self_attn.out_proj.bias_mask'])
            parameters_to_prune.append(model.encoder.layers[ii].fc1)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.fc1.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.fc1.bias_mask'])
            parameters_to_prune.append(model.encoder.layers[ii].fc2)
            mask_list_w.append(mask_dict['encoder.layers.' + str(ii) + '.fc2.weight_mask'])
            mask_list_b.append(mask_dict['encoder.layers.' + str(ii) + '.fc2.bias_mask'])

    #for ii in range(7): # only applying weight mask
    #    prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list_w[ii])
    #for ii in range(7, len(parameters_to_prune)): # applying both weight+bias masks
    #    prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list_w[ii])
    #    prune.CustomFromMask.apply(parameters_to_prune[ii], 'bias', mask=mask_list_b[ii-7])
    for ii in range(0, len(parameters_to_prune)): # applying both weight+bias masks
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list_w[ii])
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'bias', mask=mask_list_b[ii])


