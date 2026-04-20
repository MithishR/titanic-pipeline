y_true=[1,0,1,1,0,1,0,0,1,0]
y_pred=[1,0,0,1,0,1,1,0,1,1]
def get_counts(y_true, y_pred):
    tp=sum((y_true[i]==1 and y_pred[i]==1) for i in range(len(y_true)))
    fp=sum((y_true[i]==0 and y_pred[i]==1) for i in range(len(y_true)))
    tn=sum((y_true[i]==0 and y_pred[i]==0) for i in range(len(y_true)))
    fn=sum((y_true[i]==1 and y_pred[i]==0) for i in range(len(y_true)))
    return tp, fp, tn, fn
def precision(y_true,y_pred):
    tp,fp,tn,fn=get_counts(y_true,y_pred)
    return tp/(tp+fp) if (tp+fp)>0 else 0
def recall(y_true,y_pred):
    tp,fp,tn,fn=get_counts(y_true,y_pred)
    return tp/(tp+fn) if (tp+fn)>0 else 0

def f1_score(y_true,y_pred):
    precision_score=precision(y_true,y_pred)
    recall_score=recall(y_true,y_pred)
    return 2*(precision_score*recall_score)/(precision_score+recall_score) if (precision_score+recall_score)>0 else 0

def cosine_similarity(A,B):
    dot_product=np.dot(A,B)
    magnitude_A=np.linalg.norm(A)
    magnitude_B=np.linalg.norm(B)
    return dot_product/(magnitude_A*magnitude_B) if magnitude_A>0 and magnitude_B>0 else 0

tp,fp,tn,fn=get_counts(y_true,y_pred)
prec=precision(y_true,y_pred)
rec=recall(y_true,y_pred)
f1=f1_score(y_true,y_pred)
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
print(f'Precision: {prec:.2f}, Recall: {rec:.2f}, F1 Score: {f1:.2f}')

from sklearn.metrics import precision_score, recall_score, f1_score
print(f"sklearn - Precision: {precision_score(y_true, y_pred):.2f}")
print(f"sklearn - Recall: {recall_score(y_true, y_pred):.2f}")
print(f"sklearn - F1: {f1_score(y_true, y_pred):.2f}")