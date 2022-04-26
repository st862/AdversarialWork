import CycleGAN_datasets
import torch
import torch.optim as optim
from CycleGAN_helper_functions import Load_Models, Final_loss, Plots
from Models import Linear

def main():

    user_input = False

    if user_input:
        test_id = input ("Enter Test ID: ")
        assert test_id in CycleGAN_datasets.list(), "Test ID not recognised"
    else:
        test_id = "test1"
        assert test_id in CycleGAN_datasets.list(), "Test ID not recognised"

    Test = CycleGAN_datasets.dictionary()[test_id]
    
    encoder_shape = Test.encoder_shape
    decoder_shape = Test.decoder_shape
    Encoder_A = Linear(encoder_shape,"relu")
    Decoder_A = Linear(decoder_shape,"relu")
    Encoder_B = Linear(encoder_shape,"relu")
    Decoder_B = Linear(decoder_shape,"relu")

    path = Test.path

    Encoder_A,Decoder_A = Load_Models(Encoder_A,Decoder_A,path+"_Enc_A",path+"_Dec_A")
    Encoder_B,Decoder_B = Load_Models(Encoder_B,Decoder_B,path+"_Trained_Enc",path+"_Trained_Dec")
    
    Final_loss(Encoder_A,Decoder_B,Encoder_B,Decoder_B,Test.dataloader_test_A,Test.dataloader_test_B,Test.shift)
    Plots(Encoder_A,Decoder_B,Encoder_B,Decoder_B,Test.dataloader_test_A,Test.dataloader_test_B,Test.shift)

main()
