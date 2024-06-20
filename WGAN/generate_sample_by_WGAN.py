from keras.models import load_model
from sklearn.manifold import TSNE
import util

# load dataset
datasetr = util.load_real_samples('path_to_dataset.csv')
# gen dataset dga base CHARBOT and data_real
datasetf = util.load_fake_sample(datasetr)

# load black-box detector
generator = load_model('path_to_model')

# generate and print
g = load_model('path_save_generator')
noise = np.random.normal(0, 1, (100, wgan.latent_dim))
base_imgs = util.generate_fake_samples(datasetf, 100)
base_imgs = np.asarray(base_imgs)
gen_imgs = g.predict([base_imgs,noise])

#generate and using t-SNE to show
imgs = util.generate_real_samples(datasetr, 5000)
noise = np.random.normal(0, 1, (5000, wgan.latent_dim))
im = util.generate_fake_samples(datasetf, 5000)
base_imgs = []
for i in im:
  temp = array2domain(i)
  temp = util.generate_dga_domain(temp)
  temp, mv = domain2array(temp)
  temp = temp.reshape((25,40,1))
  base_imgs.append(temp)
base_imgs = np.asarray(base_imgs)
gen_imgs = generator.predict([base_imgs,noise])
imgs = np.asarray(imgs)
X = np.vstack((gen_imgs,imgs))
X = np.reshape(X,[X.shape[0],X.shape[1]*X.shape[2]*X.shape[3]]) 

tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(X)
y = np.concatenate((np.ones((5000)),np.zeros((5000))))
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df).set(title="MNIST data T-SNE projection")