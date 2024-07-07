import numpy as np

def generate_synthetic_data():

    num_train = 1400
    num_val_test = 280
    nsteps = 5
    x_dim = 8
    a_dim = 1
    r_dim = 1

  
    x_train = np.random.randn(num_train, nsteps, x_dim).astype(np.float32)
    a_train = np.random.randint(0, 2, size=(num_train, nsteps, a_dim)).astype(np.float32) 
    r_train = np.random.randn(num_train, nsteps, r_dim).astype(np.float32)  
    mask_train = np.ones((num_train, nsteps, 1), dtype=np.float32)  


    x_validation = np.random.randn(num_val_test, nsteps, x_dim).astype(np.float32)
    a_validation = np.random.randint(0, 2, size=(num_val_test, nsteps, a_dim)).astype(np.float32)
    r_validation = np.random.randn(num_val_test, nsteps, r_dim).astype(np.float32)
    mask_validation = np.ones((num_val_test, nsteps, 1), dtype=np.float32)


    x_test = np.random.randn(num_val_test, nsteps, x_dim).astype(np.float32)
    a_test = np.random.randint(0, 2, size=(num_val_test, nsteps, a_dim)).astype(np.float32)
    r_test = np.random.randn(num_val_test, nsteps, r_dim).astype(np.float32)
    mask_test = np.ones((num_val_test, nsteps, 1), dtype=np.float32)


    np.savez('synthetic_gaussian_train.npz', x_train=x_train, a_train=a_train, r_train=r_train, mask_train=mask_train)
    np.savez('synthetic_gaussian_validation.npz', x_validation=x_validation, a_validation=a_validation, r_validation=r_validation, mask_validation=mask_validation)
    np.savez('synthetic_gaussian_test.npz', x_test=x_test, a_test=a_test, r_test=r_test, mask_test=mask_test)

    print("Synthetic data has been generated and saved.")

if __name__ == "__main__":
    generate_synthetic_data()
