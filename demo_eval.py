import pandas as pd
import numpy as np 
import ast 
import tokenization
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
import warnings
from vocabulary import Vocabulary
warnings.filterwarnings("ignore")


# Reading input data 
arg_list = ast.literal_eval(str(sys.argv))
test =  pd.read_csv(arg_list[1])

#parameters
sen_len  = 22
vocab_size = 250 
latent_dim_encoder = 160
latent_dim_decoder = 160




'''
 Function to pre-process data : normalisation by removing all variables and replacing it by v1,v2, etc.
 Comments, strings characters are also replaced and added to end of row in pandas to at new column

'''
def pre_process(data):
  tokenizer = tokenization.C_Tokenizer()
  data['src_id_var'] = ""
  data['src_var_id'] = ""
  data['src_lc_s'] = ""
  data['src_lc_c']= ""
  data['src_lc_n']= ""
  data['src_lc_comm']= ""
  if 'targetLineTokens' in data.columns:
    data['tar_id_var'] = ""
    data['tar_var_id'] = ""
    data['tar_lc_s'] = ""
    data['tar_lc_c']= ""
    data['tar_lc_n']= ""
    data['tar_lc_comm']= ""


  for index in data.index:
    src_line = ast.literal_eval(data['sourceLineTokens'][index])
    line = ' '.join(src_line)
    toks,types = tokenizer.tokenize(line)
    num_var = list(set([toks[i] for i in range(len(toks)) if types[i]=='name']))
    var_num = {num_var[i]:i for i in range(len(num_var))}
    data['src_id_var'][index] = num_var
    data['src_var_id'][index] = var_num
    s_list=[]
    c_list=[]
    n_list=[]
    com_list=[]
    chk = var_num.keys();

    for i,s in enumerate(toks):
      if s in chk:
        toks[i]='v'+str(var_num[s])
      if types[i] == 'string':
        s_list.append(toks[i])
        toks[i]='lc_s'
      if types[i] == 'char':
        c_list.append(toks[i])
        toks[i]='lc_c'
      if types[i] == 'number':
        n_list.append(toks[i])
        toks[i]='lc_n'
      if types[i] == 'comment':
        com_list.append(toks[i])
        toks[i] = 'lc_comm'
    
    normalised= ['SOS']
    normalised.extend([toks[x] for x in range(min(sen_len-2,len(toks)))])
    normalised.extend(['PAD' for x in range(sen_len -1 - len(normalised))])
    normalised.append('EOS')
     
    data['sourceLineTokens'][index] = normalised
    data['src_lc_s'][index] = s_list
    data['src_lc_c'][index]= c_list
    data['src_lc_n'][index]= n_list
    data['src_lc_comm'][index]= com_list
      
    
    if 'targetLineTokens' in data.columns:
      tar_line = ast.literal_eval(data['targetLineTokens'][index])
      line = ' '.join(tar_line)
      toks,types = tokenizer.tokenize(line)
      
      num_var = list(set([toks[i] for i in range(len(toks)) if types[i]=='name']))
      var_num = {num_var[i]:i for i in range(len(num_var))}
      data['tar_id_var'][index] = num_var
      data['tar_var_id'][index] = var_num
      s_list=[]
      c_list=[]
      n_list=[]
      com_list=[]
      chk = var_num.keys();
      for i,s in enumerate(toks):
        if s in chk:
          toks[i]='v'+str(var_num[s])
        if types[i] == 'string':
          s_list.append(toks[i])
          toks[i]='lc_s'
        if types[i] == 'char':
          c_list.append(toks[i])
          toks[i]='lc_c'
        if types[i] == 'number':
          n_list.append(toks[i])
          toks[i]='lc_n'
        if types[i] == 'comment':
          com_list.append(toks[i])
          toks[i] = 'lc_comm'
      
      
      
      normalised= ['SOS']
      normalised.extend([toks[x] for x in range(min(sen_len-2,len(toks)))])
      normalised.extend(['PAD' for x in range(sen_len -1 - len(normalised))])
      normalised.append('EOS')
     
      data['targetLineTokens'][index] = normalised
      data['tar_lc_s'][index] = s_list
      data['tar_lc_c'][index]= c_list
      data['tar_lc_n'][index]= n_list
      data['tar_lc_comm'][index]= com_list
    
    
print("Pre-processing starts")
pre_process(test)
print("Pre-processing Ends")

print("\n\nDone with pre-process!\n\n")
file_to_read = open("en_vocab.pickle", "rb")
encoder_vocab = pickle.load(file_to_read)
file_to_read = open("dec_vocab.pickle", "rb")
decoder_vocab = pickle.load(file_to_read)

print("Done with vacabulory")

print('Encoder vocab word count : ',encoder_vocab.num_words)
print('Encoder vocab longest sentence ',encoder_vocab.longest_sentence)

encoder_input_data = np.zeros(
    (len(test),encoder_vocab.longest_sentence, encoder_vocab.num_words), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(test), encoder_vocab.longest_sentence, encoder_vocab.num_words), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(test), encoder_vocab.longest_sentence, encoder_vocab.num_words), dtype="float32"
)


print("Generating encoder data")

for i, (input_text, target_text) in enumerate(zip(test['sourceLineTokens'],test['targetLineTokens'])):
    for t, char in enumerate(input_text):
      if char in encoder_vocab.word2index:
        encoder_input_data[i, t, encoder_vocab.word2index[char]] = 1.0
      else:
        encoder_input_data[i, t, encoder_vocab.word2index['OOV_Token']] = 1.0
    
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        if char in encoder_vocab.word2index:
          decoder_input_data[i, t, encoder_vocab.word2index[char]] = 1.0
        else:
          decoder_input_data[i, t, encoder_vocab.word2index['OOV_Token']] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            if char in encoder_vocab.word2index:
              decoder_target_data[i, t-1, encoder_vocab.word2index[char]] = 1.0
            else:
              decoder_target_data[i, t-1, encoder_vocab.word2index['OOV_Token']] = 1.0
              
              

test_orig = pd.read_csv(arg_list[1])
tokenizer = tokenization.C_Tokenizer()

for index in test_orig.index: 
  tar_line = ast.literal_eval(test_orig['targetLineTokens'][index])
  line = ' '.join(tar_line)
  toks,types = tokenizer.tokenize(line)
  test_orig['targetLineTokens'][index] = toks



model = keras.models.load_model("third.h5")


print("\n Model loaded\n\n")
encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim_encoder,)) #input 3
decoder_state_input_c = keras.Input(shape=(latent_dim_encoder,)) # input 4
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, encoder_vocab.num_words))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, encoder_vocab.word2index['SOS']] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ["SOS"]
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = encoder_vocab.index2word[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "EOS" or len(decoded_sentence) > decoder_vocab.longest_sentence:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1,  encoder_vocab.num_words))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence


"""
procedure to de-normalise to lines 
"""
def de_normalise(toks,index,data):
  #Removing SOS ,EOS , OOV 
  c=s=com=n=0
  toks = [i for i in toks if i not in ('SOS','EOS','PAD')]
  
  for i,x in enumerate(toks):
    if (x[0]=='v' and x[1:].isdigit()):
        id= int(x[1:])
        if len(data['src_id_var'][index]) > 0:
          toks[i] = data['src_id_var'][index][id%(len(data['src_id_var'][index]))]
        continue
    if x == 'lc_s' and s < len(data['src_lc_s'][index]) :
      toks[i]=data['src_lc_s'][index][s]
      s+=1
      continue
    if x == 'lc_c' and c < len(data['src_lc_c'][index]):
      toks[i]=data['src_lc_c'][index][c]
      c+=1
      continue
    if x == 'lc_n' and n < len(data['src_lc_n'][index]):
      toks[i]=data['src_lc_n'][index][n]
      n+=1
      continue
    if x == 'lc_com' and com < len(data['src_lc_com'][index]):
      toks[i]=data['src_lc_com'][index][com]
      com+=1
      continue
    
  return toks
      
# <-------------------------------------------------------------------------------------->
print("\n\Finding output sequences n\n")
correct = 0
samples = len(test)
test_orig['fixedTokens'] = ""
for index in range(samples):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[index : index + 1]
    decoded_sentence = decode_sequence(input_seq)
    decoded_sentence = de_normalise(decoded_sentence,index,test)
    test_orig['fixedTokens'][index]=decoded_sentence
    if decoded_sentence == test_orig['targetLineTokens'][index]:
      correct+=1
    if index % 10 == 0 and index != 0: 
      print('accuracy: after',index,': ',correct/index)

print('Accuracy : ',correct/samples)
    
test_orig.to_csv(arg_list[2])



      
