A RNN appliction for Predicting The National Stock Exchange of India Limited Nifty price.

Model is trained over 3795 data of Nifty price from 1/04/2001 to 31/03/2017 and tested over 1/04/2017 to 31/03/2018 for each working day of NSE, and also for same data.

It will predict mood of market for upcoming next-week.
Here:-    Nifty_prediction_result.png- is result when test over same as training set.

![alt text](https://raw.githubusercontent.com/junior-g/Useful_Scripts_DuringDeepLearning/master/RNN-Application(Nifty-Prediction)/Nifty_prediction_result.png)

and      Nifty_prediction_result_2.png :- is when tested over test data of 1/04/2017 to 31/03/2018 (256 working days of NSE) 
![alt text](https://raw.githubusercontent.com/junior-g/Useful_Scripts_DuringDeepLearning/master/RNN-Application(Nifty-Prediction)/Nifty_prediction_result_2.png)


(This difference is due to large difference between price during 2001-2010 and 2011-2018. But still shape is very similar)

Means is accuracy is about ~94% which is very good :)

                -: Enjoy DeepLearning :) :-
