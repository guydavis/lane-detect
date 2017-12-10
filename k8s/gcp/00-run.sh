# Modified from https://github.com/mappedinn/kubernetes-nfs-volume-on-gke

# create a GCE persistent disk
gcloud compute disks create --size=200GB gke-nfs-disk

# create a GKE cluster
gcloud container clusters create gke-lane-detect --num-nodes=3

# create a NFS server in the cluster
kubectl create -f 01-dep-nfs.yml 

# create the NFS service to expose to rest of cluster
kubectl create -f 02-srv-nfs.yml

# create a NFS persistent volume and PV claim for it
nfs_ip=$(kubectl get services | grep nfs-server | tr -s ' ' | cut -d ' ' -f 3)
sed "s/NFS_IP/$nfs_ip/g" 03-pv-and-pvc-nfs.yml > 03-pv-and-pvc-nfs-replaced.yml
kubectl create -f 03-pv-and-pvc-nfs-replaced.yml

# upload local video files to the NFS share
nfs_pod=$(kubectl get pods | grep nfs-server | cut -d ' ' -f 1)
kubectl cp ../../videos $nfs_pod:/exports/videos

# create instances to process videos
kubectl create -f 04-dep-lane-detect.yml

# Check on the state of one
ld_pod=$(kubectl get pods | grep lane-detect | cut -d ' ' -f 1)
kubectl exec $ld_pod  -- du -hsc /mnt/output
