import torch
import Models
import CycleGAN_helper_functions
import CycleGAN_datasets
import matplotlib.pyplot as plt

def Test_Models():
    print("\nTesting Models.py")
    print("\nInitialising linear model with relu and sigmoid...")
    model = Models.Linear([8,4,1],"relu","sigmoid")
    print(model)
    print("checking model.forward")
    x = torch.rand(8)
    print(model(x))

def Test_CycleGAN_helper_functions():
    print("\nTesting CycleGAN_helper_functions.py")
    print("\nTesting loss functions:")
    x_0 = torch.ones(100)
    x_1 = torch.zeros(100)
    x_0_5 = 0.5 * torch.ones(100)
    x_rand = torch.rand(100)
    print("real = ones, fake = zeros")
    print(f"Loss_D: {CycleGAN_helper_functions.LOSS_D(x_0,x_1)}, Loss_G: {CycleGAN_helper_functions.LOSS_G(x_1)}")
    print("real = zeros, fake = ones")
    print(f"Loss_D: {CycleGAN_helper_functions.LOSS_D(x_1,x_0)}, Loss_G: {CycleGAN_helper_functions.LOSS_G(x_0)}")
    print("real = 0.5, fake = 0.5")
    print(f"Loss_D: {CycleGAN_helper_functions.LOSS_D(x_0_5,x_0_5)}, Loss_G: {CycleGAN_helper_functions.LOSS_G(x_0_5)}")
    print("real = rand, fake = rand")
    print(f"Loss_D: {CycleGAN_helper_functions.LOSS_D(x_rand,x_rand)}, Loss_G: {CycleGAN_helper_functions.LOSS_G(x_rand)}")
    print("Assuming the input is [0-1], minimising loss_G: fake -> 1, minimising loss_D: fake -> 0 and real -> 1 ")

def Test_CycleGAN_datasets():
    print("\nTesting CycleGAN_datasets.py")
    print("\nTest1")
    Test = CycleGAN_datasets.Test1()
    for i in Test.data_A.columns:
        plt.subplot(2,4,i+1)
        plt.hist(Test.data_A[i],alpha=0.5,bins=50,density=True)
        plt.hist(Test.data_B[i],alpha=0.5,bins=50,density=True)
    plt.show()

Test_Models()
Test_CycleGAN_helper_functions()
Test_CycleGAN_datasets()

