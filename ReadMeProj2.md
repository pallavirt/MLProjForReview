# Project: Operationalizing Machine Learning : Documentation

This project uses the same Bank Marketing dataset. Using this dataset, the best model was found by running an automated ML experiment and it was later published as a REST endpoint.

## Architectural Diagram
![image](https://user-images.githubusercontent.com/109726862/181511798-2170ca5d-142a-4efd-bf74-d11492718be8.png)

## Key Steps
Following steps were followed in Azure ML Studio using the lab provided by Udacity.
1. Create a dataset
 ![image](https://user-images.githubusercontent.com/109726862/181511921-4f055897-9c32-4d41-bc1b-840766d52787.png)
2. Use AutoML to train and find the best model for this dataset
  ![image](https://user-images.githubusercontent.com/109726862/181512310-cbd60148-334b-41a2-81b9-f38029e5e8bf.png)
3. Ensure the status is completed
  ![image](https://user-images.githubusercontent.com/109726862/181512584-8925a1a2-6a45-49b8-98c4-bda52804706e.png)
4. Observe the algorithm for the best model and deploy that as a REST endpoint
  ![image](https://user-images.githubusercontent.com/109726862/181513072-2f068eb8-d225-4248-b5f9-2298d7c4d3c8.png)
5. Enable Application Insights for the new endpoint
  ![image](https://user-images.githubusercontent.com/109726862/181513371-23cacf10-e445-4806-b5bd-74b65b08c472.png)
6. Output after running logs.py script
  ![image](https://user-images.githubusercontent.com/109726862/181513601-0c12b53c-f7a0-4b7d-bd98-7b3d8a793671.png)
7. Swagger showing endpoint APIs and parameters
  ![image](https://user-images.githubusercontent.com/109726862/181513982-8f92c05e-1921-4cb4-948f-078043742174.png)
8. Consuming the endpoint using endpoint.py script
  ![image](https://user-images.githubusercontent.com/109726862/181514305-350e7543-7308-4da6-a24e-7b233227aed1.png)
9. Pipeline job in Azure ML studio after running the given jupyter notebook
  ![image](https://user-images.githubusercontent.com/109726862/181514620-5b9dd433-4792-419a-bca1-e3ae9df5164a.png)
10. Pipeline endpoint in Azure ML Studio
  ![image](https://user-images.githubusercontent.com/109726862/181514854-849febb4-8413-4dc1-8bd6-22d87a0742d3.png)
11. Published Pipeline overview
  ![image](https://user-images.githubusercontent.com/109726862/181515139-81d73c25-8bd5-4cfb-bfad-19a93aafc0df.png)

## Screen Recording
Link to a screen recording of the project in action:
https://youtu.be/L9eoxlX82dY
