## How to deploy

1. **Fork this repo**

     Forking a repository allows you to create a personal copy of someone else’s project. This is especially useful when you want to make changes to the project without affecting the original repository or just create your own instance.

    - In the top-right corner of the page, click the **"Fork"** button. This will create a copy of the repository under your own GitHub account.
    - Once the fork is complete, you will be redirected to the newly forked repository in your GitHub account.


2. **Setup Google Cloud Project and Billing**

    - First, go to [Create a Project](https://console.cloud.google.com/projectcreate?inv=1&invt=Abxo6A) make sure to be log in in the account you want to have the project

    - Provide all the info and click Create. If you get lost, you can [check this](https://developers.google.com/workspace/guides/create-project)

    - To setup the payment you'll need to add a Credit/Debit Card. Keep in mind that Google gives you 300$ in Credits, so, as long as you dont do something wrong there's nothing to worry about

3. **Setup Cloud Run**

    [Cloud Run](https://cloud.google.com/run) is serverless tool of GCP that allows you to create both functions and container-based servers. Cloud run handles the deployment, scaling, and traffic management, scaling your service from zero to hundreds of instances 
    
    - Once in cloud run click **connect repo** and connect your gcp account with your github account.

    - Then, you need to select the repo you fork in step 1, and click continue. 

    - Now go to the security tab and enable **Unauthenticated invocations**(This is importat so that your URL accesable from your web browser). [Docs](https://cloud.google.com/run/docs/authenticating/public?hl=en)

    - Once the deploy its done you can see the Cloud Run Service's URL. You can use this URL to access to your application

## If you got a problem

1. Ask ChatGPT
2. Google it 
3. Let me know at lapiceroazul@proton.me
