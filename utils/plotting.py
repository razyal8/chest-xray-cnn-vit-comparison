import matplotlib.pyplot as plt

def plot_curves(history, out_png):
    # history: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(history['train_acc'], label='train_acc')
    if 'val_acc' in history and history['val_acc']:
        plt.plot(history['val_acc'], label='val_acc')
    plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_acc.png"))
    plt.close()
