import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import random 
import seaborn as sns

def LOSS_D(real,fake):
    return torch.mean((real-1)**2) + torch.mean(fake**2)

def LOSS_G(fake):
    return torch.mean((fake-1)**2)

def Epoch_D(Encoder_A,Decoder_A,Encoder_B,Decoder_B,Discrim_A,Discrim_B,optimizer_D_A,optimizer_D_B,dataloader_A,dataloader_B,device):
    for (X_A,X_B) in zip(dataloader_A,dataloader_B):
        with torch.no_grad():
            a_real = X_A.to(device)
            b_real = X_B.to(device)
            b2a = Decoder_A(Encoder_B(b_real))
            a2b = Decoder_B(Encoder_A(a_real))
            
        optimizer_D_A.zero_grad()
        Disc_loss_A = LOSS_D (Discrim_A(a_real),Discrim_A(b2a))
        Disc_loss_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        Disc_loss_B = LOSS_D (Discrim_B(b_real),Discrim_B(a2b.detach()))
        Disc_loss_B.backward()
        optimizer_D_B.step()

def Epoch_G(Encoder_A,Decoder_A,Encoder_B,Decoder_B,Discrim_A,Discrim_B,optimizer_G,dataloader_A,dataloader_B,device,C_ae,C_cyc,C_disc):
    for (X_A,X_B) in zip(dataloader_A,dataloader_B):
        a_real = X_A.to(device)
        b_real = X_B.to(device)
        a_reconstructed=Decoder_A(Encoder_A(a_real))
        b_reconstructed=Decoder_B(Encoder_B(b_real))
        b2a = Decoder_A(Encoder_B(b_real))
        a2b = Decoder_B(Encoder_A(a_real))
        a_cycle = Decoder_A(Encoder_B(a2b))
        b_cycle = Decoder_B(Encoder_A(b2a))
        
        fool_loss_A = LOSS_G(Discrim_A(b2a))
        fool_loss_B = LOSS_G(Discrim_B(a2b))
        AE_loss_A = nn.L1Loss()(a_real,a_reconstructed)
        AE_loss_B = nn.L1Loss()(b_real,b_reconstructed)  
        cycle_loss_A = nn.L1Loss()(a_real,a_cycle)
        cycle_loss_B = nn.L1Loss()(b_real,b_cycle)
        optimizer_G.zero_grad()
        Loss = C_ae * AE_loss_A + C_ae * AE_loss_B + \
            C_cyc * cycle_loss_A + C_cyc * cycle_loss_B + \
            C_disc * fool_loss_A + C_disc * fool_loss_B
        Loss.backward()
        optimizer_G.step()
    print(f"Losses - AE: {AE_loss_A:.3f}/{AE_loss_B:.3f} fool: {fool_loss_A:.3f}/{fool_loss_B:.3f}")


def Save_Models(enc,dec,path_e,path_d):
    torch.save(enc.state_dict(), path_e)
    torch.save(dec.state_dict(), path_d)

def Load_Models(enc,dec,path_e,path_d):
    enc.load_state_dict(torch.load(path_e,map_location=torch.device('cpu')))
    dec.load_state_dict(torch.load(path_d,map_location=torch.device('cpu')))
    return enc,dec

def Train_AE(Encoder,Decoder,dataloader,dataloader_test,device,lr=1e-2):
    optimizer=optim.Adam([{'params':Encoder.parameters()},{'params':Decoder.parameters()}], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    for epoch in range(10000):
        for X in dataloader:
            optimizer.zero_grad()
            X=X.to(device)
            X_reconstructed=Decoder(Encoder(X))
            loss=nn.MSELoss()(X_reconstructed,X)
            loss.backward()
            optimizer.step()
        for X in dataloader_test:
            X=X.to(device)
            X_reconstructed=Decoder(Encoder(X))
            val_loss=nn.MSELoss()(X_reconstructed,X)
            scheduler.step(val_loss)
        if epoch % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Loss {loss.item()}, LR {lr}")
            if lr<1e-7:
                return Encoder,Decoder
    return Encoder,Decoder

def Final_loss (Encoder_A,Decoder_A,Encoder_B,Decoder_B,dataloader_A,dataloader_B,shift_A2B):

    for (X_A,X_B) in zip(dataloader_A,dataloader_B):
        a_real=X_A
        b_real=X_B
        a_reconstructed=Decoder_A(Encoder_A(a_real))
        b_reconstructed=Decoder_B(Encoder_B(b_real))
        b2a = Decoder_A(Encoder_B(b_real))
        a2b = Decoder_B(Encoder_A(a_real))
        a2b_man = a_real + torch.tensor(shift_A2B)
        b2a_man = b_real - torch.tensor(shift_A2B)
        AE_loss_A = nn.L1Loss()(a_real,a_reconstructed)
        AE_loss_B = nn.L1Loss()(b_real,b_reconstructed)                    
        a2b_loss_A = nn.L1Loss()(a2b,a2b_man)
        a2b_loss_B = nn.L1Loss()(b2a,b2a_man)
        break
    print(f"Reconstruction Loss: {AE_loss_A,AE_loss_B}, A2B Loss: {a2b_loss_A,a2b_loss_B}")

def Plots(ENCODER_A,DECODER_A,ENCODER_B,DECODER_B,dataloader_A,dataloader_B,shift_A2B):
    with torch.no_grad():
        for X_A in dataloader_A:
            Z_A=ENCODER_A(X_A)
            X_A_rec=DECODER_A(ENCODER_A(X_A))
            X_A2B = DECODER_B(ENCODER_A(X_A))
            X_A2B_manually = X_A + torch.tensor(shift_A2B)
            X_A2B_man_encoded = ENCODER_B(X_A2B_manually)
        for X_B in dataloader_B:
            Z_B=ENCODER_B(X_B)
            X_B_rec=DECODER_B(ENCODER_B(X_B))
            X_B2A=DECODER_A(ENCODER_B(X_B))
            X_B2A_manually = X_B - torch.tensor(shift_A2B)
            X_B2A_man_encoded = ENCODER_A(X_B2A_manually)
    print("Reconstructions")

    i = random.randint(1,len(X_A))
    plt.plot(X_A[i],label="X")
    plt.plot(X_A_rec[i],label="D_x(E_x(X))",linestyle = ":")
    plt.title("One Example of X vs D_x(E_x(X)).")
    plt.legend()
    plt.show()

    i = random.randint(1,len(X_B))
    plt.plot(X_B[i],label="Y")
    plt.plot(X_B_rec[i],label="D_y(E_y(Y))",linestyle = ":")
    plt.title("One Example of Y vs D_y(E_y(Y)).")
    plt.legend()
    plt.show()

    A_rec_error = X_A - X_A_rec
    sns.boxplot(data = A_rec_error)
    plt.title("X - D_x(E_x(X)) For X in Machine A")
    plt.show()

    B_rec_error = X_B - X_B_rec
    sns.boxplot(data = B_rec_error)
    plt.title("Y - D_y(E_y(Y)) For Y in Machine B")
    plt.show()

    print("Latent")

    A_latent_error = Z_A - X_A2B_man_encoded
    plt.ylim(-1,1 )
    plt.title("E_x(X) - E_y(Y*) For X in Machine A")
    sns.boxplot(data = A_latent_error)
    plt.show()

    B_latent_error = Z_B- X_B2A_man_encoded
    plt.ylim(-1,1 )
    plt.title("E_y(Y) - E_x(X*) For Y in Machine A")
    sns.boxplot(data = B_latent_error)
    plt.show()

    print("X2Y")

    i = random.randint(1,len(X_A))
    plt.plot(X_A[i],label="A",alpha=0.5)
    plt.plot(X_A2B_manually[i],label="A2B_manually",alpha=0.5)
    plt.plot(X_A2B[i],label="A2B_model",alpha=0.5)
    plt.legend()
    plt.title("X vs Y* vs D_y(E_x(X)) for X in Machine A")
    plt.show()

    i = random.randint(1,len(X_B))
    plt.plot(X_B[i],label="B",alpha=0.5)
    plt.plot(X_B2A_manually[i],label="B2A_manually",alpha=0.5)
    plt.plot(X_B2A[i],label="B2A",alpha=0.5)
    plt.title("Y vs X* vs D_x(E_y(Y)) for Y in Machine B")
    plt.legend()
    plt.show()

    A2B_error = X_A2B - X_A2B_manually
    plt.title("Y* - D_y(E_x(X)) for X in Machine A")    
    sns.boxplot(data = A2B_error)
    plt.show()

    B_rec_error = X_B2A - X_B2A_manually
    plt.title("X* - D_y(E_x(Y)) for Y in Machine B")
    sns.boxplot(data = B_rec_error)
    plt.show()