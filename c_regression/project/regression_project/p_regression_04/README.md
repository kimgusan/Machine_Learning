# ① Regression01

## 주제: # ☄️ 주제: 쌍성 데이터 확인

    (1) 데이터 원본: ex) https://kaggle.com)

### 목차

1. **가설 설정**
2. **데이터 분석**
3. **데이터 전처리**
4. **데이터 훈련**
    - Cycle (n)반복 - 각 이미지에 대하여 링크를 걸어서 확인할 것.
5. **결론**

## 1. 가설 설정

## 가설 설정

### 주제: 쌍성 식별

#### 가설: 별의 쌍성 여부를 정확하게 분석할 수 있는가?

-   **가설 내용**:
    -   천문학 데이터를 통해 별의 쌍성 여부를 예측할 수 있을지 검토합니다. 특정 천문학적 요소들이 별이 쌍성인지 아닌지를 정확하게 식별하는 데 중요한 역할을 할 것으로 예상됩니다.
    -   이 가설은 천문학적 데이터 분석을 통해 별의 쌍성 여부를 정확하게 식별하고, 관련 요소를 파악하여 천문학적 연구에 기여할 수 있다는 가정에 기반합니다.

#### 분석 목표:

-   **천문 데이터를 사용하여 쌍성인 별을 식별**:
    -   천문 데이터에 포함된 다양한 속성(예: 별의 밝기, 위치, 운동 속도 등)을 분석하여 쌍성 여부를 예측합니다.
    -   이러한 데이터를 사용하여 쌍성을 구성하는 별들 간의 상호 작용과 그 특성을 이해하고자 합니다.

#### 예상 결과:

-   분석을 통해 쌍성 별을 식별하는 주요 요소를 파악하고, 이를 통해 더 정확한 천문학적 예측 모델을 개발할 수 있습니다.
-   이 연구는 별의 쌍성 여부를 보다 정확하게 식별하는 방법론을 제시하며, 이는 천문학적 연구와 관측에 중요한 정보를 제공할 것입니다.

<hr>

### 2. 데이터 분석

```
import pandas as pd

train = pd.read_csv('../../datasets/p_cross_train.csv')
test = pd.read_csv('../../datasets/p_cross_test.csv')

s_df = pd.concat([train, test])
s_df

```

### 3. 데이터 전처리

```
# 결측치 확인
s_df.isna().sum()

# 중복값 확인
s_df.duplicated().sum()
```

### 4. 데이터 훈련

-   Cycle01
    1. 별도의 전처리 없이 훈련 진행.

```
s_df.is_binary.value_counts()
- 비중 그래프 첨부

s_df.corr()['is_binary'].sort_values(ascending=False)[1:]
- 상관관계 표 첨부
- 상관관계 히트맵 첨부
- OLS 지표 첨부
```

```
# 로지스틱 회귀 모델 훈련
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

features, targets = s_df.iloc[:, :-1],s_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, stratify=targets, random_state=321)

lg = LogisticRegression(solver='liblinear', random_state=321, max_iter=10000)
# lg.fit(over_X_train, over_y_train)
lg.fit(X_train, y_train)
prediction = lg.predict(X_test)
get_evaluation(y_test, prediction, lg, X_test)

- 평가지표 그래프 첨부
```

-   Cycle02
    1. 언더샘플링 진행 후 로지스틱 회귀 분석 진행

```
# 언더샘플링
sl0 = s_df[s_df['is_binary']==0].sample(2999, random_state=321)
sl1 = s_df[s_df['is_binary']==1]

under_s_df = pd.concat([sl0, sl1]).reset_index(drop=True)
under_s_df['is_binary'].value_counts()

- 타겟 비중 같아진 그래프 첨부
- OLS 지표 첨부
```

```
# 로지스틱 회귀 모델 훈련
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

features, targets = under_s_df.iloc[:, :-1],under_s_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, stratify=targets, random_state=321)

lg = LogisticRegression(solver='liblinear', random_state=321, max_iter=10000)
# lg.fit(over_X_train, over_y_train)
lg.fit(X_train, y_train)
prediction = lg.predict(X_test)

get_evaluation(y_test, prediction, lg, X_test)
- 평가지표 그래프 첨부
```

-   Cycle03
    1. 모델의 수치들에 대하여 표준화를 위해 Power transform 진행

```
# Power Trnsform 진행
from sklearn.preprocessing import PowerTransformer

columns = under_s_df.iloc[:, :-1].columns
p_u_s_df = under_s_df.copy()

for column in columns:
    ptf = PowerTransformer(standardize=False)
    result = ptf.fit_transform(under_s_df[[column]])
    p_u_s_df[column] = result

p_u_s_df

- 히스토그램 그래프 before 첨부
- 히스토그램 그래프 after 첨부 첨부
```

```
- OLS 지표 그래프 첨부
```

```
# 로지스틱 회귀 모델 훈련
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

features, targets = p_u_s_df.iloc[:, :-1],p_u_s_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, stratify=targets, random_state=321)

lg = LogisticRegression(solver='liblinear', random_state=321, max_iter=10000)
# lg.fit(over_X_train, over_y_train)
lg.fit(X_train, y_train)
prediction = lg.predict(X_test)

get_evaluation(y_test, prediction, lg, X_test)
- 평가지표 그래프 첨부
```

-   Cycle04
    1. 점수가 높게 나왔으며 과적합을 대비하기 위한 교차검증 진행

```
from sklearn.model_selection import cross_val_score, KFold

features, targets = p_u_s_df.iloc[:,:-1], p_u_s_df.iloc[:,-1]

kf = KFold(n_splits=10, random_state=321, shuffle=True)
scores = cross_val_score(lg, features, targets , cv=kf)
scores

array([0.92833333, 0.93833333, 0.91833333, 0.93166667, 0.94333333,
       0.95166667, 0.93333333, 0.93166667, 0.92320534, 0.92988314])
```

```
# 차원축소 진행
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 데이터 불러오기
features, targets = p_u_s_df.iloc[:,:-1], p_u_s_df.iloc[:,-1]

# 데이터 분할
X_train, X_test, y_train, y_test =\
train_test_split(features, targets, test_size=0.2, random_state=321)

# K-폴드 교차 검증 설정
kfold = KFold(n_splits=15, random_state=321, shuffle=True)

# 로지스틱 회귀 모델 설정
lg = LogisticRegression(solver='liblinear', random_state=321, max_iter=10000)

# 파라미터 그리드 설정
parameters = {
    'C': [0.1]
}

# GridSearchCV 설정
grid_lgb = GridSearchCV(lg, param_grid=parameters, cv=kfold, scoring='accuracy')  # 'r2' 대신 'accuracy' 사용
grid_lgb.fit(X_train, y_train)

- 평가 지표 그래프 첨부
```

-   Cycle05
    1. 다중공선성을 가진 feature 제거

```
- OLS 지표 첨부
- 다중공선성 표 첨부
pre_s_df = p_u_s_df.drop(labels=['b'], axis = 1)
pre_s_df

pre_s_df = p_u_s_df.drop(labels=['b'], axis = 1)
pre_s_df
- 특정 feature 삭제
```

```
## from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

features, targets = pre_s_df.iloc[:,:-1], pre_s_df.iloc[:,-1]

X_train, X_test, y_train, y_test = \
train_test_split(features,targets, test_size=0.2, random_state=321)

# K-폴드 교차 검증 설정
kfold = KFold(n_splits=15, random_state=321, shuffle=True)

# 로지스틱 회귀 모델 설정
lg = LogisticRegression(solver='liblinear', random_state=321, max_iter=10000)

# 파라미터 그리드 설정
parameters = {
    'C': [1]
    # 'penalty': ['l1', 'l2']
}

# GridSearchCV 설정
grid_lgb = GridSearchCV(lg, param_grid=parameters, cv=kfold, scoring='accuracy')  # 'r2' 대신 'accuracy' 사용
grid_lgb.fit(X_train, y_train)

- 평가지표 그래프 첨부

```

-   Cycle06
    1. 하이퍼파라미터 조정이 명확하지 않아 파이토치를 사용하여 모델 훈련

```
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from torch.optim import SGD
from sklearn.model_selection import train_test_split

torch.manual_seed(321)

features, targets = pre_s_df.iloc[:, :-1], pre_s_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, random_state=124)

X_train = torch.FloatTensor(X_train.values)
y_train = torch.FloatTensor(y_train.values).view(-1, 1)
X_test = torch.FloatTensor(X_test.values)
y_test = torch.FloatTensor(y_test.values).view(-1, 1)

W = torch.zeros((6, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = SGD([W, b], lr=0.000004)

# 반복 횟수
epochs = 10000

for epoch in range(1, epochs + 1):
    # 가설 선언
    # H = 1 / (1 + torch.exp(-(X_train.matmul(W) + b)))
    H = torch.sigmoid(X_train.matmul(W) + b)

    # 손실 함수 선언
    # losses = -(y_train * torch.log(H) + (1 - y_train) * torch.log(1 - H))
    # loss = losses.mean()
    loss= binary_cross_entropy(H, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 100 epoch 단위로 로그 출력
    if epoch % 1000 == 0:
        print('{:4d}/{}: W1: {:.4f}, W2: {:.4f}, W3: {:.4f}, b: {:.4f}, loss: {:.4f}'\
              .format(epoch, epochs, W[0].item(), W[1].item(), W[2].item(), b.item(), loss.item()))

get_evaluation(y_test.detach().numpy(), torch.sigmoid(X_test.matmul(W) + b) >= 0.5)

- 평가지표 표 확인
```

```
#import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from torch.optim import SGD

torch.manual_seed(321)

features, targets = pre_s_df.iloc[:, :-1], pre_s_df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
train_test_split(features, targets, test_size=0.2, random_state=124)

X_train = torch.FloatTensor(X_train.values)
y_train = torch.FloatTensor(y_train.values).view(-1, 1)
X_test = torch.FloatTensor(X_test.values)
y_test = torch.FloatTensor(y_test.values).view(-1, 1)

logistic_r = nn.Sequential(
   nn.Linear(6, 1), # 결과 출력
   nn.Sigmoid() # 출력시 시그모이드 함수를 통과한다.
)
optimizer = SGD(logistic_r.parameters(), lr=4e-6)

# 반복 횟수
epochs = 100000

for epoch in range(1, epochs + 1):
    # 가설 선언
    H = logistic_r(X_train)

    # 손실 함수 선언
    loss= binary_cross_entropy(H, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10000 == 0:
        print(f'Epoch: {epoch}/{epochs},', end=' ')
        for i, w in enumerate(list(logistic_r.parameters())[0][0]):
            print(f'W{i + 1}: {np.round(w.item(), 4)}', end=', ')
        print(f'b: {np.round(list(logistic_r.parameters())[1].item())} Loss: {np.round(loss.item(), 4)}')

get_evaluation(y_test.detach().numpy(), logistic_r(X_test) >= 0.5)

- 평가지표 표 첨부
```

```
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from torch.optim import SGD
import matplotlib.pyplot as plt

torch.manual_seed(321)

# 데이터 불러오기와 전처리
features, targets = pre_s_df.iloc[:, :-1], pre_s_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=124)
X_train = torch.FloatTensor(X_train.values)
y_train = torch.FloatTensor(y_train.values).view(-1, 1)
X_test = torch.FloatTensor(X_test.values)
y_test = torch.FloatTensor(y_test.values).view(-1, 1)

# 모델 구성
logistic_r = nn.Sequential(
   nn.Linear(6, 1),
   nn.Sigmoid()
)
optimizer = SGD(logistic_r.parameters(), lr=4e-6)

# 반복 횟수
epochs = 300000
real_train_loss_history = []
val_test_loss_history = []

for epoch in range(1, epochs + 1):
    # 가설 선언
    H = logistic_r(X_train)

    # 손실 함수 선언
    loss = binary_cross_entropy(H, y_train)
    real_train_loss_history.append(loss.item())

    # 검증 손실 계산
    with torch.no_grad():
        val_loss = binary_cross_entropy(logistic_r(X_test), y_test)
        val_test_loss_history.append(val_loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10000 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')

# 손실 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(real_train_loss_history, label='Training Loss', linewidth=3, color='skyblue')
plt.plot(val_test_loss_history, label='Validation Loss', linestyle='--', linewidth=3, alpha=0.3, color='red')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.title('Loss Trend Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

- 손실함수 그래프 첨부

get_evaluation(y_test.detach().numpy(), logistic_r(X_test) >= 0.5)

- 평가지표 표 첨부
```

```
# 임계치 확인
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_recall_curve
import numpy as np

def precision_recall_curve_plot(y_test , prediction_proba_class1):

    precisions, recalls, thresholds = precision_recall_curve(y_test, prediction_proba_class1)

    # X축: threshold
    # Y축: 정밀도, 재현율
    # 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')

    # X축(threshold)의 Scale을 0 ~ 1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))

    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.grid()
    plt.show()

# precision_recall_curve_plot(y_test, lg.predict_proba(X_test)[:, 1] )

import numpy as np
precision_recall_curve_plot(y_test, grid_lgb.predict_proba(X_test)[:, 1] )

- AUC-ROC 그래프 확인

```

```
# 임계치별 평가 지표 확인
_, _, thresholds = precision_recall_curve(y_test, grid_lgb.predict_proba(X_test)[:, 1])
thresholds

from sklearn.preprocessing import Binarizer
def get_evaluation_by_thresholds(y_test, prediction_proba_class1, thresholds):

    for threshold in thresholds:
        binarizer = Binarizer(threshold=threshold).fit(prediction_proba_class1)
        custom_prediction = binarizer.transform(prediction_proba_class1)
        print('임곗값:', threshold)
        get_evaluation(y_test, custom_prediction)

get_evaluation_by_thresholds(y_test, grid_lgb.predict_proba(X_test)[:, 1].reshape(-1, 1), thresholds)

- 임계치, 평가지표 그래프 첨부
```

```
# 임계치 수정후 그래프 작성을 위한 평가지표 함수
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# 타겟 데이터와 예측 객체를 전달받는다.
def get_evaluation_by_custom_proba(y_test, prediction, visualize=False):
#     오차 행렬
    confusion = confusion_matrix(y_test, prediction)
#     정확도
    accuracy = accuracy_score(y_test , prediction)
#     정밀도
    precision = precision_score(y_test , prediction)
#     재현율
    recall = recall_score(y_test , prediction)
#     F1 score
    f1 = f1_score(y_test, prediction)
#     ROC-AUC
    roc_auc = roc_auc_score(y_test, prediction)

    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, ROC-AUC: {4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
    print("#" * 80)

    if visualize:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
        titles_options = [("Confusion matrix", None), ("Normalized confusion matrix", "true")]

        for (title, normalize), ax in zip(titles_options, axes.flatten()):
            disp = ConfusionMatrixDisplay.from_predictions(y_test, prediction, ax=ax, cmap=plt.cm.Blues, normalize=normalize)
            disp.ax_.set_title(title)
        plt.show()

binarizer = Binarizer(threshold=0.8558796444911337)
custom_prediction = binarizer.fit_transform(grid_lgb.predict_proba(X_test)[:, 1].reshape(-1, 1))
get_evaluation_by_custom_proba(y_test, custom_prediction, True)

- 평가지표 그래프 첨부
from sklearn.metrics import roc_curve

def roc_curve_plot(y_test , custom_proba):
#     임계값에 따른 FPR, TPR 값
    fprs, tprs, thresholds = roc_curve(y_test, custom_proba)

#     ROC Curve를 plot 곡선으로 그림.
    plt.plot(fprs , tprs, label='ROC')
#     가운데 대각선 직선을 그림.
#     TPR과 FPR이 동일한 비율로 떨어진다는 것은 모델이 양성과 음성을 구별하지 못한다는 것을 의미한다.
#     다른 분류기를 판단하기 위한 기준선으로 사용되며,
#     대각선에 가까울 수록 예측에 사용하기 힘든 모델이라는 뜻이다.
    plt.plot([0, 1], [0, 1], 'k--', label='Standard')

    # X축(FPR)의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )'); plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()

- 임계치 수정 후 Roc Curve 곡선 첨부
```

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 준비
data = {
    'Cycle': ['Cycle01', 'Cycle02', 'Cycle03', 'Cycle04', 'Cycle05', 'Cycle06a', 'Cycle06b', 'Cycle06c'],
    'Accuracy': [0.9913, 0.9367, 0.9358, 0.4942, 0.4942, 0.6733, 0.7000, 0.8983],
    'Precision': [0.4954, 0.9471, 0.9532, 0.5078, 0.5078, 0.7227, 0.7521, 0.9665],
    'Recall': [0.1800, 0.9250, 0.9167, 0.4756, 0.4756, 0.5554, 0.5906, 0.8238],
    'F1': [0.2641, 0.9359, 0.9346, 0.4912, 0.4912, 0.6281, 0.6617, 0.8895],
    'AUC': [0.5892, 0.9367, 0.9358, 0.4947, 0.4947, 0.6726, 0.6993, 0.8978]
}
df = pd.DataFrame(data)

# 데이터를 'Cycle'로 그룹화하여 플롯
df_melted = df.melt(id_vars=['Cycle'], var_name='Metric', value_name='Value')

# 막대 그래프 그리기
plt.figure(figsize=(14, 8))
bar_plot = sns.barplot(x='Cycle', y='Value', hue='Metric', data=df_melted, palette='viridis')

# 각 막대에 수치 추가
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.2f'),
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha = 'center', va = 'bottom',
                      size=10,
                      xytext = (0, 10),
                      textcoords = 'offset points',
                      rotation=45)

plt.title('Performance Metrics by Cycle')
plt.ylabel('Score')
plt.xlabel('Cycle')
plt.xticks(rotation=45)
plt.ylim(0, 1.05)  # 점수 범위를 [0, 1]로 설정
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


- 수치에 대한 bar 그래프 첨부
```

## 결론

### 분석 결과

-   **성능 개선 조치**: 언더샘플링과 임계치 조정을 통해 쌍성 별을 식별하는 데 필요한 정밀도를 최대화하였습니다. 이 접근법은 쌍성 별의 정확한 식별에 크게 기여하였습니다.
-   **훈련 방법**: 정밀도를 중시하는 모델 구축을 위해 적절한 데이터 전처리 후 체계적인 모델 훈련을 수행하였습니다. 결과적으로, 훈련된 모델은 실제 쌍성 별을 효과적으로 감지하는 능력을 보였습니다.

### 기술적 접근

-   **모델의 특징**: 언더샘플링 기법을 사용하여 훈련 데이터의 균형을 맞추고, 정밀도가 높은 예측 모델을 구현했습니다. 이 모델은 쌍성 별을 정확하게 분류하는 데 중요한 요소들을 고려하였습니다.
-   **기술적 적용**: 사용된 모델은 고도의 데이터 분석 기법을 적용하여 복잡한 데이터 구조에서 유의미한 정보를 추출하였습니다. 이러한 접근은 과적합을 방지하고 모델의 일반화 능력을 향상시켰습니다.

### 응용 가능성

-   **과학적 연구와 관측**: 이 연구 결과는 쌍성 별의 정확한 식별을 통해 천문학적 연구에 필수적인 정보를 제공합니다. 이는 별의 질량, 궤도, 발광과 같은 중요한 특성을 이해하는 데 도움이 됩니다.
-   **천문학 데이터의 활용**: 이 모델은 천문학 데이터의 복잡성을 효과적으로 처리하며, 실제 관측에서 쌍성 별을 정확하게 감지하는 능력을 강화합니다.

이 결론은 프로젝트의 성공을 강조하며, 사용된 방법론과 기술이 어떻게 실제 응용에 기여할 수 있는지를 설명합니다.