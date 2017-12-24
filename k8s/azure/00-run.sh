# Documents CLI steps to create a k8s cluster on Azure

# First create a "Resource Group" to hold the cluster
az group create --location centralus --name guy-azure

# Then can create the k8s cluster in that group
az aks create -g guy-azure -n lane-detect --kubernetes-version 1.8.1

# Create an Azure file share and upload the source videos
current_env_conn_string=$(az storage account show-connection-string -n guydavisazure -g guy-azure --query 'connectionString' -o tsv)
if [[ $current_env_conn_string == "" ]]; then  
    echo "Couldn't retrieve the connection string."
    exit 1
fi
az storage share create --name myazurefiles --quota 200 --connection-string $current_env_conn_string 
# Manually upload dashcam footage via Azure Console in your browser.

# Get the base64 of Azure Storage Account name and access key, put into this yml script, then create k8s secret
kubectl create -f 01-create-secret.yml 

# create the persistent volume and claim
kubectl create -f 02-create-pv.yml
kubectl create -f 03-create-pvc.yml

# create 3 instances to process the videos
kubectl create -f 04-dep-lane-detect.yml

# Check on the state of one to see processed video
ld_pod=$(kubectl get pods | grep lane-detect | cut -d ' ' -f 1)
kubectl exec $ld_pod  -- du -hsc /mnt/output
