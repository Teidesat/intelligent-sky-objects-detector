import requests
import json
import os


default_url = 'http://nova.astrometry.net'
fits_file_url = default_url+'/new_fits_file/'
axy_file_url = default_url+'/axy_file/'
job_status_url = default_url+'/api/jobs/'
#api_key = 'mmwzqqcoweoercyp'
dataset_path = './dataset'
job_base_range = 1000000
jobs_range = range(job_base_range, job_base_range + 1000)
job_succesful = "{\"status\": \"success\"}"
os.makedirs(dataset_path, exist_ok=True)
# Login with your API key to get a session key
#R = requests.post(default_url+'/login', data={'request-json': json.dumps({"apikey": api_key})})

for id in jobs_range:
  print(id)
  try:
    status = requests.get(job_status_url+str(id))
    if status.content.decode() == job_succesful:
      img = requests.get(fits_file_url+str(id))
      axy = requests.get(axy_file_url+str(id))
      image_folder = os.path.join(dataset_path, str(id))
      #print(image_folder)
      img_type = img.headers.get("Content-Type", "")
      axy_type = axy.headers.get("Content-Type", "")
      if img.status_code == 200 and img_type == 'application/fits' and axy_type == 'application/fits':
      #Extract filename from URL
        os.makedirs(image_folder, exist_ok=True)
        img_name = os.path.join(image_folder, f"{id}.fits")
        img_axy = os.path.join(image_folder, f"{id}axy.fits")
        with open(img_name, "wb") as file:
          file.write(img.content)  
          print(f"Image saved as {img_name}")
        with open(img_axy, "wb") as file:
          file.write(axy.content)  
          print(f"Image data saved as {img_axy}")
      else:
        print("Failed to download image")
    else:
        print("Job was not successful")
  except:
    print("Failed to download image")
  #test = os.path.join(fits_file_url, str(id))
  #print(test)
  
#print(R.text)
#status = requests.get(job_status_url+str(12609030))
#img = requests.get(axy_file_url+'12609020')
#content_type = img.headers.get("Content-Type", "")
#print(status)
#print(status.content.decode())
#print("{\"status\": \"success\"}")
#print(status.content.decode() == "{\"status\": \"success\"}")
#exit()
#image_folder = os.path.join(dataset_path, '12609032')
#os.makedirs(image_folder, exist_ok=True)
#filename = os.path.join(image_folder, f"12609032.fits")
#with open(filename, "wb") as file:
#  file.write(img.content)