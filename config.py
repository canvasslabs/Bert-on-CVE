config = {
    'MAX_LEN' : 200, #max length of a sequence
    'bs' : 16, #batch size
    'substitue' : 'X', #substitute tag
    'name' : 'bert-base-uncased', #model name
    'do_lower_case' : True, #lower case
    'if_cross_val' : True, #if do cross validation or not
    'fold_num' : 5, #number of fold to cross validation
    'test_size' : 0, #ratio of test set to the whole data when if_cross_val==False. if 0, then wouldn't test
    'hidden_size' : 768, #input size of classifier
    'dropout' : 0.1, #dropout rate
    'decay' : 1e-3, #weight decay
    'lr' : 8e-5, #learning rate
    'mu' : 1, #parameter in computing weights
    'beta' : 2, #ratio of recall over precision
    'FULL_FINETUNING' : True, #if finetune bert or not
    'max_grad_norm' : 1.0, #grad clipping
    'epochs' : 50, #training epochs
    'period' : 100, #period of display training results
}
