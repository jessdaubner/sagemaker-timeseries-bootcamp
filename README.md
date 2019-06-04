# Predict the Future with Time Series Forecasting and Deep Neural Networks
This collection of resources is a companion to the Predict the Future with Time Series Forecasting and Deep Neural Networks bootcamp.

# AWS Resources and References
Start here to learn more about Amazon Sagemaker to Build, Train, and Deploy Machine Learning Models at Scale. Then check out more resources below.
* Amazon Sagemaker Developer Guide summarizes the service and frameworks https://docs.aws.amazon.com/sagemaker/index.html#lang/en_us
* Access more than 30 Amazon Sagemaker Notebooks at https://github.com/awslabs/amazon-sagemaker-examples
* New (circa 27Mar2019)– AWS Deep Learning Containers https://aws.amazon.com/blogs/aws/new-aws-deep-learning-containers/
* Amazon QuickSight Announces General Availability of ML Insights (circa Mar2019) https://aws.amazon.com/blogs/big-data/amazon-quicksight-announces-general-availability-of-ml-insights/
* Overview of containers for Amazon SageMaker https://sagemaker-workshop.com/custom/containers.html
* Find prebuilt ML and DL models on the AWS Marketplace https://aws.amazon.com/marketplace/solutions/machinelearning/

# Deep Learning Resources
* Dive into Deep Learning, An interactive deep learning book with code, math, and discussions that is authored and maintained by Amazon AI specialists. https://d2l.ai/. Think textbook and Jupyter notebooks integrated in a learning portal.
* MIT Online Deep Learning book / labs https://www.deeplearningbook.org/ This is a good read for anyone wanting a refresher or first broad look at Deep Learning.

# Time Series Forecasting with Neural Networks (module 1)
* For an introduction to Recurrent Neural Networks (RNNs) start with Understanding LSTM Networks, http://colah.github.io/posts/2015-08-Understanding-LSTMs/ then read The Unreasonable Effectiveness of Recurrent Neural Networks https://karpathy.github.io/2015/05/21/rnn-effectiveness/
* Blog post on developing LSTM networks in Python using the Keras deep learning library for time series forecasting. https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
* Keras documentation on recurrent layers. https://keras.io/layers/recurrent/
* The Unreasonable Effectiveness of Recurrent Neural Networks, a blog post on RNNs by Andrej Karpathy. http://karpathy.github.io/2015/05/21/rnn-effectiveness/
* Research paper cited in Module 1 to illustrate an example of non-linear, complex dependencies. Carbonneau, Réal, Rustam Vahidov, and Kevin Laframboise. “Forecasting Supply Chain Demand Using Machine Learning Algorithms.” Machine Learning, 1652–1686. doi:10.4018/978-1-60960-818-7.ch609. https://pdfs.semanticscholar.org/85c8/66eeeaf05edf872e7d06ff1c32f7fd560757.pdf (Abstract only)
* Deutsche Börse Public Dataset with links to the datasets and alternative visualizations and machine learning examples. https://registry.opendata.aws/deutsche-boerse-pds/
  - Stock Price Movement Prediction Using The Deutsche Börse Public Dataset & Machine Learning is a docker based ML share price predictor https://github.com/Originate/dbg-pds-tensorflow-demo
  - 10 visualizations to try in Amazon QuickSight with sample data https://aws.amazon.com/blogs/big-data/10-visualizations-to-try-in-amazon-quicksight-with-sample-data/
* Efficient Backprop by Yann Le Cunn describes strategies for improving or minimizing the cost function. You can find it and many other interesting ML and DL approaches at http://yann.lecun.com/exdb/publis/index.html
* Predicting stock prices is non trivial. Here is an untested approach using GANs which looks interesting to say the least. Titled 'Using the latest advancements in AI to predict stock market movements' https://github.com/borisbanushev/stockpredictionai

# Amazon SageMaker and DeepAR for Time Series Forecasting (module 2)
* The seminal research paper on DeepAR. DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Network https://pdfs.semanticscholar.org/4eeb/e0d12aefeedf3ca85256bc8aa3b4292d47d9.pdf
* Latest publication (as of March, 2019) related to and building off of DeepAR. Deep State Space Models for Time Series Forecasting https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf
* DeepAR Forecasting documentation https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html
* Time series forecasting with DeepAR This notebook demonstrates how to prepare a data set of time series for training DeepAR and how to use the trained model for inference. https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/deepar_synthetic/deepar_synthetic.ipynb
* SageMaker/DeepAR demo on electricity dataset This notebook complements the notebook cited above. In this notebook, a real-life use case will be considered to show you how to use DeepAR on SageMaker for predicting energy consumption. This is the notebook from which Lab 2 of this bootcamp was designed. https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/deepar_electricity/DeepAR-Electricity.ipynb

# Introduction to MXNet and Gluon (module 3)
* Deep Learning – The Straight Dope A repo containing an incremental sequence of notebooks designed to teach deep learning, Apache MXNet, and the Gluon interface. https://gluon.mxnet.io/ and simoncorstonoliver/DeepLearningWithMXNetGluon forked from ThomasDelteil/DeepLearningWithMXNetGluon https://github.com/simoncorstonoliver/DeepLearningWithMXNetGluon
* Machine Learning with Time Series and Sequence Data, This repo includes instructional materials for the “Modeling timeseries and sequence data on AWS using Apache MXNet and Gluon” workshop run during the Applied ML Days, 2018. The workshop focused around how to model generic sequence data and time series data to do sequence prediction, forecasting future time series, and anomaly detection. It demonstrates how to train neural networks with Apache MXNet. https://github.com/sunilmallya/timeseries
* Pixm takes on phishing attacks with deep learning using Apache MXNet on AWS, One of two MXNet case studies highlighted in this module. https://aws.amazon.com/blogs/machine-learning/pixm-takes-on-phishing-attacks-with-deep-learning-using-apache-mxnet-on-aws/
* Curalate makes social sell with AI using Apache MXNet on AWS https://aws.amazon.com/blogs/machine-learning/curalate-makes-social-sell-with-ai-using-apache-mxnet-on-aws/

# LSTNet (module 4)
* Modeling Long and Short-Term Temporal Patterns with Deep Neural Networks, Seminal paper on LSTNet. https://arxiv.org/pdf/1703.07015.pdf
* LSTNet implementation in Apache MXNet Gluon, Github repository that this module’s lab is based off of. https://github.com/safrooze/LSTNet-Gluon
* A short primer on ReLU and Softmax Activation Functions https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions

# AWS Machine Learning Resources
* AWS has now released free digital ML training (more than 45 hours) and announced the new Machine Learning Certification https://aws.amazon.com/blogs/machine-learning/amazons-own-machine-learning-university-now-available-to-all-developers/ and https://aws.amazon.com/training/learning-paths/machine-learning/
* AWS created solutions, reference architectures and quickstarts at https://aws.amazon.com/big-data/getting-started/tutorials/ . Look for the self paced labs that are accessible at run.qwiklabs.com
* There are many interesting solutions on the AWS Machine Learning blog at https://aws.amazon.com/blogs/machine-learning/
* Access hundreds of ML algorithms and models from the AWS Marketplae https://aws.amazon.com/about-aws/whats-new/2018/11/awsmarketplace-makes-it-easier-to-build-machine-learning-applications-on-amazonsagemaker/
* Which services are available in which regions? https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/
* ReInvent 2018 was a huge event for ML. 13 major ML annoucements by AWS. https://www.businesswire.com/news/home/20181128005660/en/Amazon-Web-Services-Announces-13-New-Machine and https://aws.amazon.com/new/reinvent/?sc_icampaign=event_reinvent2018-recap&sc_ichannel=ha&sc_icontent=awssm-1690-default-hero&sc_iplace=hero&trk=ha_awssm-1690-default-hero
* 8 Machine Learning Algorithms explained in Human language (sort of) https://www.datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/
* AWS provided ML resources on Github. https://github.com/aws-samples/machine-learning-samples which includes useful tools and the comprehensive Amazon Sagemaker collection including algorithms https://github.com/awslabs/amazon-sagemaker-examples

# Other References
* List of mathematical symbols for those who need a refresher. Useful when reading the academic papers referenced here. https://en.wikipedia.org/wiki/List_of_mathematical_symbols
* Online derivative calculator using math symbols https://www.derivative-calculator.net/
* Comparing forecasting methods. https://otexts.com/fpp2/accuracy.html/ Ultimately it's the use of test sets that is the most conclusive validation method.

# One Line List of services
`Entry page for AWS Documentation https://docs.aws.amazon.com/index.html#lang/en_us`
` 165 services as of Mar 2019 according to Andy Jassy https://www.cnbc.com/2019/02/28/amazon-cloud-ceo-we-have-a-30-billion-run-rate-in-our-early-stages.html`

| #   | Name                                            | Description                                                                                  |
|-----|-------------------------------------------------|----------------------------------------------------------------------------------------------|
| 1   | Alexa for Business                              | Empower your Organization with Alexa                                                         |
| 2   | Amazon API Gateway                              | "Build, Deploy, and Manage APIs"                                                             |
| 3   | Amazon AppStream 2.0                            | Stream Desktop Applications Securely to a Browser                                            |
| 4   | Amazon Athena                                   | Query Data in S3 using SQL                                                                   |
| 5   | Amazon Aurora                                   | High Performance Managed Relational Database                                                 |
| 6   | Amazon Chime                                    | "Frustration-free Meetings, Video Calls, and Chat"                                           |
| 7   | Amazon Cloud Directory                          | Create Flexible Cloud-native Directories                                                     |
| 8   | Amazon CloudFront                               | Global Content Delivery Network                                                              |
| 9   | Amazon CloudSearch                              | Managed Search Service                                                                       |
| 10  | Amazon CloudWatch                               | Monitor Resources and Applications                                                           |
| 11  | Amazon Cognito                                  | Identity Management for your Apps                                                            |
| 12  | Amazon Comprehend                               | Discover Insights and Relationships in Text                                                  |
| 13  | Amazon Connect                                  | Cloud-based Contact Center                                                                   |
| 14  | Amazon DynamoDB                                 | Managed NoSQL Database                                                                       |
| 15  | Amazon EBS                                      | Block Storage for EC2                                                                        |
| 16  | Amazon EC2                                      | Virtual Servers in the Cloud                                                                 |
| 17  | Amazon EC2 Auto Scaling                         | Scale Compute Capacity to Meet Demand                                                        |
| 18  | Amazon Elastic Container (ECS) Registry               | Store and Retrieve Docker Images                                                             |
| 19  | Amazon Elastic Container Service (ECS)               | Run and Manage Docker Containers                                                             |
| 20  | Amazon Elastic Container Service for Kubernetes (EKS) | Run Managed Kubernetes on AWS                                                                |
| 21  | Amazon Elastic File System                      | Managed File Storage for EC2                                                                 |
| 22  | Amazon Elastic Transcoder                       | Easy-to-use Scalable Media Transcoding                                                       |
| 23  | Amazon ElastiCache                              | In-memory Caching System                                                                     |
| 24  | Amazon Elasticsearch Service                    | Run and Scale Elasticsearch Clusters                                                         |
| 25  | Amazon EMR                                      | Hosted Hadoop Framework                                                                      |
| 26  | Amazon FreeRTOS                                 | IoT Operating System for Microcontrollers                                                    |
| 27  | Amazon GameLift                                 | "Simple, Fast, Cost-effective Dedicated Game Server Hosting"                                 |
| 28  | Amazon Glacier                                  | Low-cost Archive Storage in the Cloud                                                        |
| 29  | Amazon GuardDuty                                | Managed Threat Detection Service                                                             |
| 30  | Amazon Inspector                                | Analyze Application Security                                                                 |
| 31  | Amazon Kinesis                                  | Work with Real-time Streaming Data                                                           |
| 32  | Amazon Kinesis Video Streams                    | Process and Analyze Video Streams                                                            |
| 33  | Amazon Lex                                      | Build Voice and Text Chatbots                                                                |
| 34  | Amazon Lightsail                                | Launch and Manage Virtual Private Servers                                                    |
| 35  | Amazon Lumberyard                               | "A Free Cross-Platform 3D Game Engine with Full Source, Integrated with AWS and Twitch"      |
| 36  | Amazon Machine Learning                         | Machine Learning for Developers                                                              |
| 37  | Amazon Macie                                    | "Discover, Classify, and Protect Your Data"                                                  |
| 38  | Amazon MQ                                       | Managed Message Broker for ActiveMQ                                                          |
| 39  | Amazon Neptune                                  | Fully Managed Graph Database Service                                                         |
| 40  | Amazon Pinpoint                                 | Push Notifications for Mobile Apps                                                           |
| 41  | Amazon Polly                                    | Turn Text into Lifelike Speech                                                               |
| 42  | Amazon Quicksight                               | Fast Business Analytics Service                                                              |
| 43  | Amazon RDS                                      | "Managed Relational Database Service for MySQL, PostgreSQL, Oracle, SQL Server, and MariaDB" |
| 44  | Amazon Redshift                                 | "Fast, Simple, Cost-effective Data Warehousing"                                              |
| 45  | Amazon Rekognition                              | Analyze Image and Video                                                                      |
| 46  | Amazon Route 53                                 | Scalable Domain Name System                                                                  |
| 47  | Amazon S3                                       | Scalable Storage in the Cloud                                                                |
| 48  | Amazon SageMaker                                | "Build, Train, and Deploy Machine Learning Models at Scale"                                  |
| 49  | Amazon Simple Email Service (SES)               | Email Sending and Receiving                                                                  |
| 50  | Amazon Simple Notification Service (SNS)        | "Pub/Sub, Mobile Push and SMS"                                                               |
| 51  | Amazon Simple Queue Service (SQS)               | Managed Message Queues                                                                       |
| 52  | Amazon Sumerian                                 | Build and Run VR and AR Applications                                                         |
| 53  | Amazon Transcribe                               | Automatic Speech Recognition                                                                 |
| 54  | Amazon Translate                                | Natural and Fluent Language Translation                                                      |
| 55  | Amazon VPC                                      | Isolated Cloud Resources                                                                     |
| 56  | Amazon WorkDocs                                 | Enterprise Storage and Sharing Service                                                       |
| 57  | Amazon WorkMail                                 | Secure and Managed Business Email and Calendaring                                            |
| 58  | Amazon WorkSpaces                               | Desktop Computing Service                                                                    |
| 59  | Apache MXNet on AWS                             | "Scalable, High-performance Deep Learning"                                                   |
| 60  | AWS Application Discovery Service               | Discover On-Premises Applications to Streamline Migration                                    |
| 61  | AWS AppSync                                     | Real-time and Offline Mobile Data Apps                                                       |
| 62  | AWS Auto Scaling                                | Scale Multiple Resources to Meet Demand                                                      |
| 63  | AWS Batch                                       | Run Batch Jobs at Any Scale                                                                  |
| 64  | AWS Certificate Manager                         | "Provision, Manage, and Deploy SSL/TLS Certificates"                                         |
| 65  | AWS Cloud9                                      | "Write, Run, and Debug Code on a Cloud IDE"                                                  |
| 66  | AWS CloudFormation                              | Create and Manage Resources with Templates                                                   |
| 67  | AWS CloudHSM                                    | Hardware-based Key Storage for Regulatory Compliance                                         |
| 68  | AWS CloudTrail                                  | Track User Activity and API Usage                                                            |
| 69  | AWS CodeBuild                                   | Build and Test Code                                                                          |
| 70  | AWS CodeCommit                                  | Store Code in Private Git Repositories                                                       |
| 71  | AWS CodeDeploy                                  | Automate Code Deployment                                                                     |
| 72  | AWS CodePipeline                                | Release Software using Continuous Delivery                                                   |
| 73  | AWS CodeStar                                    | Develop and Deploy AWS Applications                                                          |
| 74  | AWS Command Line Interface                      | Unified Tool to Manage AWS Services                                                          |
| 75  | AWS Config                                      | Track Resource Inventory and Changes                                                         |
| 76  | AWS Data Pipeline                               | "Orchestration Service for Periodic, Data-driven Workflows"                                  |
| 77  | AWS Database Migration Service                  | Migrate Databases with Minimal Downtime                                                      |
| 78  | AWS Deep Learning AMIs                          | Quickly Start Deep Learning on EC2                                                           |
| 79  | AWS DeepLens                                    | Deep Learning Enabled Video Camera                                                           |
| 80  | AWS Device Farm                                 | "Test Android, FireOS, and iOS Apps on Real Devices in the Cloud"                            |
| 81  | AWS Direct Connect                              | Dedicated Network Connection to AWS                                                          |
| 82  | AWS Directory Service                           | Host and Manage Active Directory                                                             |
| 83  | AWS Elastic Beanstalk                           | Run and Manage Web Apps                                                                      |
| 84  | AWS Elemental MediaConvert                      | Convert File-based Video Content                                                             |
| 85  | AWS Elemental MediaLive                         | Convert Live Video Content                                                                   |
| 86  | AWS Elemental MediaPackage                      | Video Origination and Packaging                                                              |
| 87  | AWS Elemental MediaStore                        | Media Storage and Simple HTTP Origin                                                         |
| 88  | AWS Elemental MediaTailor                       | Video Personalization and Monetization                                                       |
| 89  | AWS Fargate                                     | Run Containers without Managing Servers or Clusters                                          |
| 90  | AWS Glue                                        | Prepare and Load Data                                                                        |
| 91  | AWS Greengrass                                  | "Local Compute, Messaging, and Sync for Devices"                                             |
| 92  | AWS Identity and  Access Management             | Manage User Access and Encryption Keys                                                       |
| 93  | AWS IoT 1-Click                                 | One Click Creation of an AWS Lambda Trigger                                                  |
| 94  | AWS IoT Analytics                               | Analytics for IoT Devices                                                                    |
| 95  | AWS IoT Button                                  | Cloud Programmable Dash Button                                                               |
| 96  | AWS IoT Core                                    | Connect Devices to the Cloud                                                                 |
| 97  | AWS IoT Device Defender                         | Security Management for IoT devices                                                          |
| 98  | AWS IoT Device Management                       | "Onboard, Organize, and Remotely Manage IoT Devices"                                         |
| 99  | AWS Key Management Service                      | Managed Creation and Control of Encryption Keys                                              |
| 100 | AWS Lambda                                      | Run your Code in Response to Events                                                          |
| 101 | AWS Migration Hub                               | Track Migrations from a Single Place                                                         |
| 102 | AWS Mobile Hub                                  | "Build, Test, and Monitor Apps"                                                              |
| 103 | AWS Mobile SDK                                  | Mobile Software Development Kit                                                              |
| 104 | AWS OpsWorks                                    | Automate Operations with Chef and Puppet                                                     |
| 105 | AWS Organizations                               | Policy-based Management for Multiple AWS Accounts                                            |
| 106 | AWS Personal Health Dashboard                   | Personalized View of AWS Service Health                                                      |
| 107 | AWS Server Migration Service                    | Migrate On-Premises Servers to AWS                                                           |
| 108 | AWS Serverless Application Repository           | "Discover, Deploy, and Publish Serverless Applications"                                      |
| 109 | AWS Service Catalog                             | Create and Use Standardized Products                                                         |
| 110 | AWS Shield                                      | DDoS Protection                                                                              |
| 111 | AWS Single Sign-On                              | Cloud Single Sign-On (SSO) Service using SAML and a user portal for multiple accounts and applications                                                          |
| 112 | AWS Snowball                                    | Petabyte-scale Data Transport                                                                |
| 113 | AWS Snowball Edge                               | Petabyte-scale Data Transport with On-board Compute                                          |
| 114 | AWS Snowmobile                                  | Exabyte-scale Data Transport                                                                 |
| 115 | AWS Step Functions                              | Coordinate Distributed Applications                                                          |
| 116 | AWS Storage Gateway                             | Hybrid Storage Integration also available as hardware from amazon.com                                                                   |
| 117 | AWS Systems Manager                             | Gain Operational Insights and Take Action                                                    |
| 118 | AWS Trusted Advisor                             | Optimize Performance and Security                                                            |
| 119 | AWS WAF                                         | Filter Malicious Web Traffic                                                                 |
| 120 | AWS X-Ray                                       | Analyze and Debug Your Applications                                                          |
| 121 | Elastic Load Balancing                          | High Scale Load Balancing                                                                    |
| 122 | TensorFlow on AWS                               | Open-source Machine Intelligence Library                                                     |
| 123 | VMware Cloud on AWS                             | Build a Hybrid Cloud without Custom Hardware                                                 |
| 124 | Secrets Manager                             | Easily rotate, manage, and retrieve database credentials, API keys, and other secrets through their lifecycle                                                 |
| 125 | AWS Firewall Manager                             | Centrally configure and manage firewall rules across accounts and applications                                                 |
| 126 | AWS Answers (AWS Answers contains solutions and is not a service per se)                             | Clear answers to common questions about architecting, building, and running applications on the Amazon Web Services Cloud                                   |
| 127  | Amazon Kinesis Data Firehose                                  | Prepare and load real-time data streams into data stores and analytics tools
| 128  | Amazon Kinesis Data Streams                                  | Ingest and process streaming data with custom applications
| 129  | Firecracker                                  | virtualization and open source technology that enables service owners to operate secure multi-tenant container-based and serverless services by combining the speed, resource efficiency, and performance enabled by containers with the security and isolation offered by traditional VMs. MicroVM
| 130  | AWS Outposts                                  | Bring native AWS services, infrastructure, and operating models to virtually any data center, co-location space, or on-premises facility
| 131  | Amazon Robomaker                                  | develop, simulate, and deploy intelligent robotics applications at scale
| 132  | AWS DeepRacer                                  | Fully autonomous 1/18th scale race car driven by reinforcement learning, 3D racing simulator, and global racing league
| 133  | Amazon Forecast                                  | Fully managed service that uses machine learning to highly accurate forecasts
| 134  | Amazon FSx for Windows File Server                                  | Fully managed native Microsoft Windows file system that makes it easy to move Windows-based applications that require file storage to AWS
| 135  | Amazon Sagemaker NEO                                  | Train models once, and run anywhere
| 136  | Amazon Sagemaker Ground Truth                                  | Automate and crowd source the development of ML training datasets
| 137  | AWS GroundStation                                  | Communicate with satellites
| 138  | Amazon Personalize                                  | Add performant, real-time personalization and recommendations to their applications easily with no ML experience required
| 139  | Amazon TimeStream                                  | Fast, Scalable, Fully Managed Time Series Database
| 140  | Amazon Managed Streaming for Kafka                                  | Automatically provisions and runs their Apache Kafka clusters and the Apache Zookeeper nodes
| 141  | AWS Inferentia (chip)                                 | Machine learning inference chip designed to deliver high performance at low cost
| 142  | Amazon Comprehend Medical                                  | Fully managed, highly accurate deep-learning based medical NLP service
| 143  | AWS Well-Architected Tool                                  | Review the state of their workloads and compares them to the latest AWS architectural best practices
| 144  | Amazon Quantum Ledger Database (QLDB)                               | Fully managed ledger database that provides a transparent, immutable, and cryptographically verifiable transaction log ‎owned by a central trusted authority
| 145  | AWS Managed Blockchain                                  | Build and manage fully managed scalable blockchain network using open source frameworks Hyperledger Fabric and Ethereum
| 146  | AWS ThinkBox                                  | Powerful and easy to use render management system
| 147  | AWS License Manager                                  | Set rules to manage, discover, and report software license usage
| 148  | AWS App Mesh                                 | easily monitor and control communications across microservices applications.
| 149  | AWS Cloud Map                                  | Service discovery for cloud resources
| 150  | Amazon RDS on VMware                                  | run Amazon RDS managed relational databases in VMware vSphere on-premises data centers
| 151  | AWS Lake Formation                                  | Easily set up a secure data lake in days
| 152  | Amazon Sagemaker Reinforcement Learning (RL)                                  | Develop reinforcement learning models at scale
| 153  | Amazon Cloudwatch Logs Insights                                  | Fast, Interactive Log Analytics
| 154  | Amazon DynamoDB Transactions                                  | Transactions provide atomicity, consistency, isolation, and durability (ACID) in DynamoDB
| 155  | Amazon DynamoDB On-Demand                                  | Flexible capacity mode for DynamoDB capable of serving thousands of requests per second without capacity planning
| 156  | Amazon Comprehend Medical                                  | natural language processing service to extract relevant medical information from unstructured text
| 157  | AWS IoT SiteWise                                  | collect and organize your data from industrial equipment at scale
| 158  | AWS IoT Things Graph                                  | connect devices and web services to build IoT applications
| 159  | AWS Amplify Console                                  | continuous deployment and hosting service for modern web applications with serverless backends
| 160  | AWS Global Accelerator                                 | network layer service (static ip) that you can deploy in front of your internet applications to improve the availability and performance for your globally-distributed user base
| 161  | AWS Transit Gateway                                  | connect thousands of Amazon Virtual Private Clouds (VPCs) and their on-premises networks using a single gateway
| 162  | AWS IoT Events                                  | detect and respond to events from IoT sensors and applications
| 163  | AWS IoT Device Tester                                  | Windows/Linux/Mac test automation tool for connected devices
| 164  | AWS DevPay                                  | Deprecated payment service
| 165  | AWS Dynamic Training for Deep Learning                                  | open-source deep learning project that allows you to reduce model training cost and time by leveraging the cloud's elasticity and scale
| 166  | AWS Transfer for SFTP                                   | move your file transfer workloads that use the Secure Shell File Transfer Protocol (SFTP) to AWS without needing to modify your applications or manage any SFTP servers
| 167  | AWS DataSync                                  | Simplify, Automate, and Accelerate Online Data Transfer between on-premises storage and Amazon S3 or Amazon Elastic File System (Amazon EFS)
| 168  | AWS ParallelCluster                                  | Deploy and manage High Performance Computing (HPC) clusters in the AWS cloud
| 169  | AWS Resource Manager                                  | Cross account sharing capabilities on Subnets, Transit Gateways and BYOL Manager Configurations with any AWS account or through your AWS Organization
| 170  | AWS Client VPN | TLS based secure access to any resource in AWS (EC2, S3, Dynamo DB, etc.) and on-premises from anywhere using OpenVPN based clients
| 171 | Amazon Corretto | Free multiplatform and production-ready distribution of the Open Java Development Kit
| 172 | AWS Backup | Centrally manage and automate backups across AWS services including EBS, RDS, DynamoDB Tables, EFS and Storage Gateway Volumes
| 173 | Amazon DocumentDB | Fast, scalable, highly available MongoDB-compatible database
| 174 | Amazon Worklink | Provide secure mobile access to your internal websites and web apps
| 175 | AWS Resource Access Manager | AWS RAM enables you to share your resources with any AWS account or organization
| 176 | Amazon Data Lifecycle Manager | Create lifecycle policies to automate operations on specified resources
| 177 | AWS Tools for PowerShell | PowerShell modules, built on functionality exposed by the AWS SDK for .NET
| 178 | AWS CLI | Command line interface for AWS Services
| 179 | AWS SDKs | Software development kits for a wide range of languages and protocols
| 180 | AWS Silk | Amazon Fire browser Platform
| 181 | Amazon Textract | Document text and tabular content detection and analysis
| 182 | Amazon Simple Workflow Service (SWF) | Build, run, and scale background jobs that have parallel or sequential steps with state tracking and task coordination
| 183 | AWS Security Hub | Comprehensive view of your security state within AWS (Amazon GuardDuty, Amazon Inspector, Amazon Macie and AWS Config) and helps you check your compliance with the security industry standards and best practices.
| 184 | AWS Control Tower | Set up and govern your multi-account AWS environment. Uses (AWS Organizations, AWS Service Catalog, AWS Single Sign-on, AWS Config)
| 185 | aws-shell | The interactive productivity booster for the AWS CLI
| 186 | Amazon Elastic Graphics ( was Amazon EC2 Elastic GPUs ) | Easily and cost-effectively add graphics acceleration to Amazon EC2 Instances
| 187 | SimpleDB | Highly available, flexible, and scalable non-relational data store (superseded by DynamoDB but still used internally for EMR)
| 188 | AWS Pricing Calculator | Estimate the cost for your architecture solution. Configure a cost estimate that fits your unique business or personal needs with AWS products and services.
| 189 | AWS Solutions |  Vetted, technical reference implementations designed to help you solve common problems and build faster
