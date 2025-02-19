from load_cifar import get_cifar_dataset
from train_test_utils import Trainer

def main():
    batch_size = 256
    classes = 100
    train_loader, test_loader = get_cifar_dataset(batch_size=batch_size, classes=classes)
    trainer = Trainer(train_loader=train_loader, 
                      test_loader=test_loader,
                      learning_rate=1e-3,
                      epochs=100,
                      test_round=5,
                      classes=classes)
    
    trainer.train()

if __name__ == '__main__':
    main()