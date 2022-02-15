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
    parameters_to_prune_no_bias =[]
    parameters_to_prune_LSTM =[]
    parameters_to_prune_LSTMCell =[]

    # encoder 
    for ii in range(3):
        parameters_to_prune_no_bias.append((model.tts.enc.convs[ii][0]))
    parameters_to_prune_LSTM.append((model.tts.enc.blstm))

    # decoder (att)
    parameters_to_prune.append((model.tts.dec.att.mlp_enc))
    parameters_to_prune_no_bias.append((model.tts.dec.att.mlp_dec))
    parameters_to_prune_no_bias.append((model.tts.dec.att.mlp_att))
    parameters_to_prune_no_bias.append((model.tts.dec.att.loc_conv))
    parameters_to_prune.append((model.tts.dec.att.gvec))

    # decoder (lstm)
    for ii in range(2):
        parameters_to_prune_LSTMCell.append((model.tts.dec.lstm[ii].cell))

    # decoder (prenet)
    for ii in range(2):
        parameters_to_prune.append((model.tts.dec.prenet.prenet[ii][0]))

    # decoder (postnet)
    for ii in range(5):
        parameters_to_prune_no_bias.append((model.tts.dec.postnet.postnet[ii][0]))

    for ii in range(0, len(parameters_to_prune)): # applying both weight+bias masks
        prune.remove(parameters_to_prune[ii], 'weight')
        prune.remove(parameters_to_prune[ii], 'bias')
    for ii in range(0, len(parameters_to_prune_no_bias)):
        prune.remove(parameters_to_prune_no_bias[ii], 'weight')
    for ii in range(0, len(parameters_to_prune_LSTM)): # applying both weight+bias masks
        prune.remove(parameters_to_prune_LSTM[ii], 'weight_ih_l0')
        prune.remove(parameters_to_prune_LSTM[ii], 'weight_hh_l0')
        prune.remove(parameters_to_prune_LSTM[ii], 'bias_ih_l0')
        prune.remove(parameters_to_prune_LSTM[ii], 'bias_hh_l0')
        prune.remove(parameters_to_prune_LSTM[ii], 'weight_ih_l0_reverse')
        prune.remove(parameters_to_prune_LSTM[ii], 'weight_hh_l0_reverse')
        prune.remove(parameters_to_prune_LSTM[ii], 'bias_ih_l0_reverse')
        prune.remove(parameters_to_prune_LSTM[ii], 'bias_hh_l0_reverse')
    for ii in range(0, len(parameters_to_prune_LSTMCell)): # applying both weight+bias masks
        prune.remove(parameters_to_prune_LSTMCell[ii], 'weight_ih')
        prune.remove(parameters_to_prune_LSTMCell[ii], 'weight_hh')
        prune.remove(parameters_to_prune_LSTMCell[ii], 'bias_ih')
        prune.remove(parameters_to_prune_LSTMCell[ii], 'bias_hh')

def pruning_transformer_tts(model, px):
    """
    for pruning out Tacotron2 (not transformer-TTS)
    prune out Tacotron2: encoder, prenet, decoder, postnet

    note: position encoding and projection heads are not pruned. 
    """

    parameters_to_prune =[]

    # encoder 
    for ii in range(3):
        parameters_to_prune.append((model.tts.enc.convs[ii][0], 'weight'))
    parameters_to_prune.append((model.tts.enc.blstm, 'weight_ih_l0'))
    parameters_to_prune.append((model.tts.enc.blstm, 'weight_hh_l0'))
    parameters_to_prune.append((model.tts.enc.blstm, 'bias_ih_l0'))
    parameters_to_prune.append((model.tts.enc.blstm, 'bias_hh_l0'))
    parameters_to_prune.append((model.tts.enc.blstm, 'weight_ih_l0_reverse'))
    parameters_to_prune.append((model.tts.enc.blstm, 'weight_hh_l0_reverse'))
    parameters_to_prune.append((model.tts.enc.blstm, 'bias_ih_l0_reverse'))
    parameters_to_prune.append((model.tts.enc.blstm, 'bias_hh_l0_reverse'))

    # decoder (att)
    parameters_to_prune.append((model.tts.dec.att.mlp_enc, 'weight'))
    parameters_to_prune.append((model.tts.dec.att.mlp_enc, 'bias'))
    parameters_to_prune.append((model.tts.dec.att.mlp_dec, 'weight'))
    parameters_to_prune.append((model.tts.dec.att.mlp_att, 'weight'))
    parameters_to_prune.append((model.tts.dec.att.loc_conv, 'weight'))
    parameters_to_prune.append((model.tts.dec.att.gvec, 'weight'))
    parameters_to_prune.append((model.tts.dec.att.gvec, 'bias'))

    # decoder (lstm)
    for ii in range(2):
        parameters_to_prune.append((model.tts.dec.lstm[ii].cell, 'weight_ih'))
        parameters_to_prune.append((model.tts.dec.lstm[ii].cell, 'weight_hh'))
        parameters_to_prune.append((model.tts.dec.lstm[ii].cell, 'bias_ih'))
        parameters_to_prune.append((model.tts.dec.lstm[ii].cell, 'bias_hh'))

    # decoder (prenet)
    for ii in range(2):
        parameters_to_prune.append((model.tts.dec.prenet.prenet[ii][0], 'weight'))
        parameters_to_prune.append((model.tts.dec.prenet.prenet[ii][0], 'bias'))

    # decoder (postnet)
    for ii in range(5):
        parameters_to_prune.append((model.tts.dec.postnet.postnet[ii][0], 'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def see_weight_rate(model):

    # encoder
    sum_list_1, zero_sum_1 = 0, 0
    for ii in range(3):
        sum_list_1 = sum_list_1 + float(model.tts.enc.convs[ii][0].weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.enc.convs[ii][0].weight == 0))

    sum_list_1 = sum_list_1 + float(model.tts.enc.blstm.weight_ih_l0.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.enc.blstm.weight_ih_l0 == 0))
    sum_list_1 = sum_list_1 + float(model.tts.enc.blstm.weight_hh_l0.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.enc.blstm.weight_hh_l0 == 0))
    sum_list_1 = sum_list_1 + float(model.tts.enc.blstm.bias_ih_l0.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.enc.blstm.bias_ih_l0 == 0))
    sum_list_1 = sum_list_1 + float(model.tts.enc.blstm.bias_hh_l0.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.enc.blstm.bias_hh_l0 == 0))

    sum_list_1 = sum_list_1 + float(model.tts.enc.blstm.weight_ih_l0_reverse.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.enc.blstm.weight_ih_l0_reverse == 0))
    sum_list_1 = sum_list_1 + float(model.tts.enc.blstm.weight_hh_l0_reverse.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.enc.blstm.weight_hh_l0_reverse == 0))
    sum_list_1 = sum_list_1 + float(model.tts.enc.blstm.bias_ih_l0_reverse.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.enc.blstm.bias_ih_l0_reverse == 0))
    sum_list_1 = sum_list_1 + float(model.tts.enc.blstm.bias_hh_l0_reverse.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.enc.blstm.bias_hh_l0_reverse == 0))


    # decoder (att)
    sum_list_1 = sum_list_1 + float(model.tts.dec.att.mlp_enc.weight.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.att.mlp_enc.weight == 0))
    sum_list_1 = sum_list_1 + float(model.tts.dec.att.mlp_enc.bias.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.att.mlp_enc.bias == 0))

    sum_list_1 = sum_list_1 + float(model.tts.dec.att.mlp_dec.weight.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.att.mlp_dec.weight == 0))

    sum_list_1 = sum_list_1 + float(model.tts.dec.att.mlp_att.weight.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.att.mlp_att.weight == 0))

    sum_list_1 = sum_list_1 + float(model.tts.dec.att.loc_conv.weight.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.att.loc_conv.weight == 0))

    sum_list_1 = sum_list_1 + float(model.tts.dec.att.gvec.weight.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.att.gvec.weight == 0))
    sum_list_1 = sum_list_1 + float(model.tts.dec.att.gvec.bias.nelement())
    zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.att.gvec.bias == 0))

    # decoder (lstm)
    for ii in range(2):
        sum_list_1 = sum_list_1 + float(model.tts.dec.lstm[ii].cell.weight_ih.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.lstm[ii].cell.weight_ih == 0))
        sum_list_1 = sum_list_1 + float(model.tts.dec.lstm[ii].cell.weight_hh.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.lstm[ii].cell.weight_hh == 0))
        sum_list_1 = sum_list_1 + float(model.tts.dec.lstm[ii].cell.bias_ih.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.lstm[ii].cell.bias_ih == 0))
        sum_list_1 = sum_list_1 + float(model.tts.dec.lstm[ii].cell.bias_hh.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.lstm[ii].cell.bias_hh == 0))

    # decoder (prenet)
    for ii in range(2):
        sum_list_1 = sum_list_1 + float(model.tts.dec.prenet.prenet[ii][0].weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.prenet.prenet[ii][0].weight == 0))
        sum_list_1 = sum_list_1 + float(model.tts.dec.prenet.prenet[ii][0].bias.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.prenet.prenet[ii][0].bias == 0))

    # decoder (postnet)
    for ii in range(5):
        sum_list_1 = sum_list_1 + float(model.tts.dec.postnet.postnet[ii][0].weight.nelement())
        zero_sum_1 = zero_sum_1 + float(torch.sum(model.tts.dec.postnet.postnet[ii][0].weight == 0))

    total_zero_rate = 100 * zero_sum_1 / sum_list_1
    print('sum_list_1 is %d' % sum_list_1)
    print('zero_sum_1 is %d' % zero_sum_1)
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


