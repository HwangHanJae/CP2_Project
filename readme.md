# 이커머스 상품 추천 시스템

## 프로젝트 개요
- **이커머스 고객 행동데이터를 기반으로 상품 추천 시스템**을 구축하였습니다.
- **전환이 일어난 고객**, **전환이 일어나지 않은 고객으로 분류**하여 상품을 추천하였습니다.
  - 전환 : 고객이 제품을 장바구니에 담거나 구매를 한 경우
- 추천 모델로는 [implicit 라이브러리](https://github.com/benfred/implicit)를 사용했습니다.
## 프로젝트 배경 및 목적

가상의 이커머스의 데이터분석가라 가정하여 **자사의 
이커머스 비즈니스의 매출증대를 목표**로 프로젝트를 진행하였습니다.

## 데이터 출처
https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

## 데이터 분석 및 결과

해당 결과는 아래의 링크를 통하여 확인할 수 있습니다.
- [pandas_eda.ipynb](https://github.com/HwangHanJae/eCommerce-RecSystem/blob/main/EDA/pandas_eda.ipynb)
- [pandas_eda_ver2.ipynb](https://github.com/HwangHanJae/eCommerce-RecSystem/blob/main/EDA/pandas_eda_ver2.ipynb)

이커머스의 **매출증대에는 고객의 전환율, 방문자 수의 증가가 중요한 요인**입니다.

이커머스의 매출증대를 위하여 **전환이 일어나지 않은 고객, 전환이 일어난 고객으로 분류**하여 **사이트 이용률**을 확인해보았습니다.

<img width=700 alt="image1" src="https://user-images.githubusercontent.com/60374463/195752119-b4eab29e-1b65-4dec-b9c4-a7e86c595d75.png">
<!--![image](https://user-images.githubusercontent.com/60374463/195752119-b4eab29e-1b65-4dec-b9c4-a7e86c595d75.png)-->

<img width=700 alt="image1" src="https://user-images.githubusercontent.com/60374463/195752172-da5afeb5-14bf-40dd-b451-66c93512e094.png">
<!--![image](https://user-images.githubusercontent.com/60374463/195752172-da5afeb5-14bf-40dd-b451-66c93512e094.png)-->

**전환이 일어난 고객의 사이트 이용률이 그렇지 않은 고객들 보다 더 높다는 것을 확인**하였습니다.

고객들이 구매를 하지 않는 이유를 [웹 로그 분석을 통한 소비자 구매지연행동 연구 논문](https://s-space.snu.ac.kr/handle/10371/134013)에서 찾을 수 있었습니다.

| 이커머스 시장의 성장에 따라 상품이 많아지는 장점이 존재했지만  
| 상품이 많아짐에 따라 소비자가 선택해야하는 대안이 많아지고 소비자는 대안을 선택하는 과정에서 부정적인 감정을 겪게 됩니다.  
| 이때 부정적인 감정을 회피하기 위하여 구매지연행동이 나오게 됩니다.

소비자에게 좀 더 적합한 대안을 추천해 대안을 선택할 때 더 좋은 선택을 할 수 있도록 추천시스템을 구축하였습니다.


## 추천 시스템
### 최종성능
<img width=700 alt="image1" src="https://user-images.githubusercontent.com/60374463/195752222-5786207d-6873-41d6-9083-b96c2fd5b723.png">
<!--![image](https://user-images.githubusercontent.com/60374463/195752222-5786207d-6873-41d6-9083-b96c2fd5b723.png)-->

### 추천 결과

해당 결과는 [Recomendation/Recomendation_Model.ipynb](https://github.com/HwangHanJae/eCommerce-RecSystem/blob/main/Recommendation_Test/Recomdation_Model.ipynb)를 통하여 확인할 수 있습니다.

<img width=700 alt="image1" src="https://user-images.githubusercontent.com/60374463/195752232-859c520d-4a42-4c49-be78-11c2b77c9686.png">
<!--![image](https://user-images.githubusercontent.com/60374463/195752232-859c520d-4a42-4c49-be78-11c2b77c9686.png)-->
<img width=700 alt="image1" src="https://user-images.githubusercontent.com/60374463/195752241-765e5ebc-2755-4f46-9c0d-d24625b48485.png">
<!--![image](https://user-images.githubusercontent.com/60374463/195752241-765e5ebc-2755-4f46-9c0d-d24625b48485.png)-->



