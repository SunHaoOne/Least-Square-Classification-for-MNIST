# Least-Square-Classification-for-MNIST
By using Least Square Classification for MNIST, and adding random features, we finally get the 0.2 % error rate.




The major theory are  as followings:
## First

In this first experiment, we use the 493 pixel intensities, plus an additional feature with value 1, as the n = 494 features in the least squares classifier).


![](https://github.com/SunHaoOne/Least-Square-Classification-for-MNIST/blob/main/A%20matric.jpg)  

![](https://github.com/SunHaoOne/Least-Square-Classification-for-MNIST/blob/main/A.jpg)
## Second
  For the trainning dataset:
  
    We first construct the matric A (60000 by 494) and the labels(60000 by 1)
    
    By solving the equation Ax=b, we can get a least square solution ,x=inv(A.T*A)A.T*b.
    
    However, the efficiency of this method is not good enough, so we use the QR equation.
    
    The steps are as followings:
    
    1.process QR-decomposition A=QR, so you get the matric Q and R
    
    2.caculate Q.T*b
    
    3.resolve Rx=Q.T*b  
    
  For the test dataset:
  
    After we get the x matric, we caculate A_test(10000 by 494) * x(494 by 1) = b_predict(10000*1)
    
    It seems like softmax, if b>0 we reset the predict label 1 else if b<0 we reset the label -1
## Third
    Compare b_test with b_predict 
    
    Andwe can caculate TP,TN,FP,FN and precise,recall,error... 
  
## Adding stochastic features 

    Using this way can decrease the error from 1.6% to 0.2%
    
    We set a random matric R (+1 or -1)(5000 by 494), and A*R.T(60000 by 5000).
    
    Then set new A_add =[A,A*R.T],(60000 by 5494)
    
    Then we do the some things as before.
    
    Mention that when using for test_set, we should still use the random matric as before(R).
![](https://github.com/SunHaoOne/Least-Square-Classification-for-MNIST/blob/main/result.png)
    
## GPU version
    The following discription is used for CPU version and it cost much time.
    
    So I use pytorch for GPU , cuda , and @jit for C++ to accelerate the code.
    
    Finally, the running time reduced to 1/3 of the original.
    
    You can find more details in https://blog.csdn.net/qwe900/article/details/109774223
