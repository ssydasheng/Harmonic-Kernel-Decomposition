import os.path as osp
root_path = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
import rockhound as rh
import numpy as np

grid = rh.fetch_etopo1(version="bedrock")
print(grid)

def clip_closed(a, b):
    res = []
    for i in range(len(a)):
        res.append(b[np.argmin(np.abs(a[i] - b))])
    return np.asarray(res)

latitude_slice = clip_closed(list(np.array(range(-900, 900, 1)) / 10.),
                             grid.latitude.data)
longitude_slice = clip_closed(list(np.array(range(-1800, 1800, 1)) / 10.),
                              grid.longitude.data)
data = grid.sel(latitude=latitude_slice, longitude=longitude_slice)
data = data.bedrock.to_dataframe()['bedrock']
print(data)
xs = np.array([list(a) for a in data.keys().to_numpy()])
latitude = xs[:, 0]
longitude = xs[:, 1]
height = data.to_numpy()

with open(osp.join(root_path, 'data', 'ETOPO.npz'), 'wb') as file:
    np.savez(file, latitude=latitude, longitude=longitude, height=height)
