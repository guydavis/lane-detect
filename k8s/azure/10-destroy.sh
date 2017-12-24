# clean up the cluster 
kubectl delete job lane-detect 
kubectl delete pvc myazureclaim
kubectl delete pv myazurevolume

# delete the storage
current_env_conn_string=$(az storage account show-connection-string -n guydavisazure -g guy-azure --query 'connectionString' -o tsv)
if [[ $current_env_conn_string == "" ]]; then  
    echo "Couldn't retrieve the connection string."
    exit 1
fi
az storage share delete --name myazurefiles --connection-string $current_env_conn_string 

## deleting the cluster
az aks delete -g guy-azure -n lane-detect

## deleting the resource group
az group delete --name guy-azure
