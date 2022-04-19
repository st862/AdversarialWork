import CycleGAN_datasets
from CycleGAN_helper_functions import Epoch_D, Epoch_G, Save_Models, Load_Models
def main():

    user_input = True

    if user_input:
        test_id = input ("Enter Test ID: ")
        assert test_id in CycleGAN_datasets.list(), "Test ID not recognised"
        epochs, D_freq, G_freq = [ int(x) for x in input("Enter integers Epochs, D_Frequency, G_Frequency: ").split() ]
        C_ae, C_cyc, C_disc = [ float(x) for x in input ("Enter Loss Function Constants (AE, Cycle, Discriminator) : ").split() ]
    else:
        test_id = "test1"
        assert test_id in CycleGAN_datasets.list(), "Test ID not recognised"
        epochs, D_freq, G_freq = [1000, 1, 1]
        C_ae, C_cyc, C_disc = [1,1,1]
main()
