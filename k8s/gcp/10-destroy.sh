# clean up the cluster 
kubectl delete job lane-detect
kubectl delete deployment nfs-server
kubectl delete service nfs-server
kubectl delete pvc nfs
kubectl delete pv nfs

## deleting the cluster
gcloud container clusters delete gke-lane-detect

## deleting the GCE PV
gcloud compute disks delete gke-nfs-disk