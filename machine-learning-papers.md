# Questions about Machine Learning Papers

## What are some common use cases for machine learning in practical applications or research prototypes?

Kostakos and Musolesi name various use cases for machine learning. In their remarks, they focus on machine learning from an HCI perspective. In this field, activity or gesture recognition, wearable computing and thereby modeling human behavior are very common use cases for machine learning. With developing user interfaces, machine learning can help to react to users' input or even predict it. Furthermore, it also allows for optimizing the use of system resources such as batteries. Another use case, prominently features in Scully et al., is the prediction of how likely a user is to click on an article depending on factors such as the headline.

## Which problems of machine learning do the authors of the papers identify?

Kostakos et al. see the need for detailed performance evaluation of machine learning systems. Regarding generalizability to the entire population, it is essential to think about how to gather and handle train data, since there might be different user groups.

Another issue, Kostakos et al. face, is the risk of misinterpreting results of research with ML-based approaches because of its low transparency. Calculating the accuracy of a predictive machine learning model should be no real alternative to hypothesis testing but a subsequent metric. Looking at machine learning results, one cannot necessarily distinguish between causality and correlation.

They also insist on not only using accuracy itself as a measure of successful prediction. Instead, the accuracy should always be compared to a baseline. Depending on the use case, the baseline performance can already be quite high. Thus, a high value for accuracy may be less profitable than a lower value for a use case with very low baseline performance. So, besides the accuracy, it is important to take other performance measurements into account. In the domain of HCI additionally gathering qualitative data is crucial to gain an understanding of human behavior.

Sculley et al. take a more broad approach to identifying issues regarding machine learning. They do this by applying the concept of technical debt to machine learning systems. This means they analyze characteristics of software which lead to faster deployment but hinder innovation, maintenance and adaptability in the long run. They divide these issues into four categories.

Machine learning bears the risk of tight coupling and heavy dependencies of components. Seemingly simple changes to systems, like adding a new input feature for the algorithm to evaluate, can lead to unforeseen consequences on how the algorithm evaluates data as a whole. In other words, changes to one part of the system are likely to unintentionally affect other interrelated parts of the system. Sculley et al. refer to this phenomenon as _Changing Anything Changes Everything_.

In addition to the components dependencies on one another, the system as a whole is very much dependent on the input data. This might lead to problems if the model is not adapted to changes in input data. The problem might be elevated if the machine learning system evaluates features, which have little to no impact on the system's accuracy. These features increase complexity and make the system more vulnerable.

Sculley et al. also identify a tendency for machine learning software to incorporate suboptimal design patterns. One pattern is the incorporation of so-called _glue code_, which is code that is only needed to connect encapsuled machine learning modules with one another. A system can consist of 5% actual machine learning code and 95% _glue code_. Very complex data manipulation pipelines can also increase the technical debt of a software, especially if their behavior or the overall configuration is not well documented.

Lastly, as the input data is often derived from the external world, it is susceptible to change. Fixed thresholds might no longer apply, or correlations in the input data might no longer exist.

## What are the credentials of the authors with regard to machine learning? Have they published research on machine learning (or using machine-learning techniques) previously?

Kostakos and Musolesi seem to look at the topic of machine learning from an almost exclusively HCI perspective. Previous work dealing with machine learning, for example, uses this technology to investigate and enhance users' behavior when interacting with a computer – mostly a smartphone. Here, it is clearly important not only to consider metrics such as accuracy but also to ask users about their experiences. However, this does not necessarily mean that this is also beneficial for evaluating machine learning approaches in any other use cases.

Sculley et al. all work in the field of machine learning as either engineers or researches for Google. This means they are able to offer in depth and hands on insights on how to tackle problems of machine learning. Sculley et al. specifically acknowledge the influence of many colleagues on their findings.