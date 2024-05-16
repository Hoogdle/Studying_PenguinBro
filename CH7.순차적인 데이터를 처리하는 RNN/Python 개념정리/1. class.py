### 모델 코드를 보면 super()에 매개변수가 들어가는 경우가 많이 보이는데
### 클래스에 대한 개념이 부족하다고 느껴 해당 내용을 정리해보았습니다.

class A():
    def p(self):
        print('hello A')

class B(A):
    def p(self):
        print('hello B')

class C(B):
    def p(self):
        print('매개변수 없는 super')
        super().p()
        print('매개변수 클래스B와 객체로 설정한 super')
        super(B,self).p()
        print('매개변수 클래스C와 객체로 설정한 super')
        super(C,self).p()
        print('클래스 C에서의 함수')
        print('hello C')


temp = C()
temp.p()


# 매개변수 없는 super
# hello B
# 매개변수 클래스B와 객체로 설정한 super
# hello A
# 매개변수 클래스C와 객체로 설정한 super
# hello B
# 클래스 C에서의 함수
# hello C


# super({클래스이름},self) 에서
# 클래스이름을 넣으면 해당 클래스 바로 위의 부모 클래스에 접근합니다.
# 클래스 이름에 매개변수가 없는 경우도 바로 위의 부모 클래스에 접근합니다.


