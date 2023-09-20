from sklearn import metrics


def ACC(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
