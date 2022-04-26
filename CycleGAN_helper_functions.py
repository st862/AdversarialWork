import torch
import torch.optim as optim
import torch.nn as nn

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

