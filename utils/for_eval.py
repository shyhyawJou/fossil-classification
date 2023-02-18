import torch



def eval_acc(model, ds, loss_fn, device):
    model.eval()
    loss, trace = 0, 0
    n_data, n_batch = len(ds.dataset), len(ds)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(ds, 1):
            x, y = x.to(device), y.to(device)
            logit = model(x)
            loss += loss_fn(logit, y) * y.numel()
            trace += torch.sum(logit.max(1)[1] == y)
            print(f"val: {i}/{n_batch}", end='\r')
    
        loss, acc = loss / n_data, trace / n_data
    
    return loss, acc

