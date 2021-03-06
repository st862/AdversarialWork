import CycleGAN_datasets
import torch
import torch.optim as optim
from CycleGAN_helper_functions import Epoch_D, Epoch_G, Save_Models, Load_Models, Train_AE
from Models import Linear

def main(test_id="test1",epochs=10000, D_freq=1, G_freq=1,C_ae=1, C_cyc=1, C_disc=1):

    user_input = False
    
    if user_input:
        test_id = input ("Enter Test ID: ")
        assert test_id in CycleGAN_datasets.list(), "Test ID not recognised"
        epochs, D_freq, G_freq = [ int(x) for x in input("Enter integers Epochs, D_Frequency, G_Frequency: ").split() ]
        C_ae, C_cyc, C_disc = [ float(x) for x in input ("Enter Loss Function Constants (AE, Cycle, Discriminator) : ").split() ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Test = CycleGAN_datasets.dictionary()[test_id]

    encoder_shape = Test.encoder_shape
    decoder_shape = Test.decoder_shape
    Encoder_A = Linear(encoder_shape,"relu").to(device)
    Decoder_A = Linear(decoder_shape,"relu").to(device)
    Encoder_B = Linear(encoder_shape,"relu").to(device)
    Decoder_B = Linear(decoder_shape,"relu").to(device)
    Discrim_A = Linear(encoder_shape+[1],"relu").to(device)
    Discrim_B = Linear(encoder_shape+[1],"relu").to(device)
    
    path = Test.path

    try:

        Encoder_A,Decoder_A = Load_Models(Encoder_A,Decoder_A,path+"_Trained_Enc",path+"_Trained_Dec")
        Encoder_B,Decoder_B = Load_Models(Encoder_B,Decoder_B,path+"_Trained_Enc",path+"_Trained_Dec")
    
    except:
    
        print (f"No Pretrained Autoencoder at {path}, training:")
        Encoder, Decoder = Train_AE(Encoder_A,Decoder_A,Test.dataloader_train_all,Test.dataloader_test_all,device)
        Save_Models(Encoder,Decoder,path+"_Trained_Enc",path+"_Trained_Dec")

        Encoder_A,Decoder_A = Load_Models(Encoder_A,Decoder_A,path+"_Trained_Enc",path+"_Trained_Dec")
        Encoder_B,Decoder_B = Load_Models(Encoder_B,Decoder_B,path+"_Trained_Enc",path+"_Trained_Dec")

    optimizer_G = optim.Adam([  {'params':Encoder_A.parameters()}, 
                                        {'params':Decoder_A.parameters()},
                                        {'params':Encoder_B.parameters()}, 
                                        {'params':Decoder_B.parameters()}],
                                        lr=1e-4)
    optimizer_D_A =optim.Adam(Discrim_A.parameters(),lr=1e-4)
    optimizer_D_B =optim.Adam(Discrim_B.parameters(),lr=1e-4)
    
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G,patience=100)
    scheduler_D_A = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D_A,patience=100)
    scheduler_D_B = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D_B,patience=100)

    for epoch in range(0,epochs):
        if epoch % 10 == 0:
            lr = optimizer_G.param_groups[0]['lr']
            print(f"Epoch: {epoch}, lr: {lr}")
            if  lr < 1e-7:
                break
        if epoch % D_freq == 0:
            Epoch_D(Encoder_A,Decoder_A,Encoder_B,Decoder_B,Discrim_A,Discrim_B,optimizer_D_A,optimizer_D_B,Test.dataloader_train_A,Test.dataloader_train_B,device)
        if epoch % G_freq == 0:
            Loss = Epoch_G(Encoder_A,Decoder_A,Encoder_B,Decoder_B,Discrim_A,Discrim_B,optimizer_G,Test.dataloader_train_A,Test.dataloader_train_B,device,C_ae,C_cyc,C_disc)
        scheduler_G.step(Loss)
        scheduler_D_A.step(Loss)
        scheduler_D_B.step(Loss)
       
    Save_Models(Encoder_A,Decoder_A,path+"_Enc_A",path+"_Dec_A")
    Save_Models(Encoder_B,Decoder_B,path+"_Enc_B",path+"_Dec_B")
