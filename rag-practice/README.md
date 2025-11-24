## 설치 방법

#### python 설치
- 파이썬 3.11 버전 설치
pyenv install 3.11

- 3.11 버전의 python 설정
pyenv global 3.11

-파이썬 버전 확인
python --version

#### Poetry 설치
- 아래의 명령어를 실행하여 Poetry 패키지 관리 도구를 설치
pip3 install poetry==1.8.5

-파이썬 가상환경 설정
poetry shell

-파이썬 패키지 일괄 업데이트
poetry update

#### 참고
.env 파일에 개인에 맞는 API KEY 입력


[2주차] 과제
질문 1:기존 RAG방식의 한계
질문 2:RAG의 장점
질문 3:monoT5와 RankLLaMA 중 성능이 더 좋은 것은?

-청킹 방법별 차이점 분석
1.RecursiveCharacterTextSplitter:
-문자/구분자 기준으로 문자수를 chunk_size(문자 개수)로 자름

2.TokenTextSplitter:
-토크나이저를 통해 토큰으로 만든 뒤 chunk_size(토큰 개수)로 자름
-문자 기준 splitter를 쓰면 1000자가 400토큰일수도 1600토큰일수도 있음
-토큰 크기를 제어해야할 때 사용

**ex** "고양이가 공을 굴린다. 강아지가 짖는다." 

토크나이저로 토큰화
-> [ "고양", "이", "가", "공", "을",
  "굴", "린", "다", ".", 
  "강아지", "가", "짖", "는", "다", "." ]

5토큰 단위로 청킹
->청크 1 :["고양", "이", "가", "공", "을"]
->청크 2 :["굴", "린", "다", ".", "강아지"]
->청크 3 :["가", "짖", "는", "다", "."]

이후 각 청크에 대해 임베딩

-검색 결과 품질 비교
큰 차이는 아니지만 대체적으로 TokenTextSplitter 나은 품질을 출력함

-어려웠던 점과 해결 과정
1.TokenTextSplitter와 다른 스플리터들의 chunk_size의 차이 : 토큰 기준 /문자 수 기준

2.TokenTextSplitter의 청킹 -> 임베딩 과정
