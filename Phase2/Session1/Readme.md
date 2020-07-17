# Deployment of Deep Neural Network model in AWS lambda:
- Serverless framework helps to create all the integration and api gateways.
- serverless.yml contains all the information and configuration regarding the deployment.
- handler.py contains the functions used for get/post requests.
- used the serverless package requirements plugin to install the packages to lambda
- Instead of uploading all the packages which would increase the size of the package more than 250 MB used the wheel file to download and use the package after deployment.
