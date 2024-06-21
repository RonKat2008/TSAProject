from flask import Flask, render_template, send_from_directory, url_for, send_file, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import torch 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from Unet import UNet3d
from skimage.util import montage
from io import BytesIO
import tempfile
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['UPLOADED_PHOTOS_DEST'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = ['.nii']
model = UNet3d(in_channels=4, n_classes=3, n_channels=24)
model.load_state_dict(torch.load(r'UNET.pth', map_location=torch.device('cpu')))
model.eval()
map_location=torch.device('cpu')

def model_image(file):
    img = nib.load(file)
    img = np.asanyarray(img.dataobj)
    images = []
    images.append(img)
    images.append(img)
    images.append(img)
    images.append(img)
    img = np.stack(images)
    img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    return img

def mask_creation(img):
    global true_mask
    mask = model(img)
    probs = torch.sigmoid(mask)
    predictions = (probs >= .33).float()
    true_mask = predictions
    return predictions
def mask_to_nifti(predictions):
    img3D = nib.Nifti1Image(predictions.squeeze()[1].squeeze().cpu().detach().numpy(), affine=np.eye(4))
    return img3D

def display_mask(img,mask):
    img_tensor = img.squeeze()[1].cpu().detach().numpy()
    mask_tensor = mask.squeeze()[1].squeeze().cpu().detach().numpy()

    image = np.rot90(montage(img_tensor))
    mask = np.rot90(montage(mask_tensor))

    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    ax.imshow(image, cmap ='gray')
    ax.imshow(np.ma.masked_where(mask == False, mask),
            cmap='cool', alpha=0.6)

    num  = 0
    for file in os.listdir():
        num += 1
    file_name ="D:\TSAProjects\static\images\\prediction" + str(num) +".png"
    plt.savefig(file_name)
    return "prediction" + str(num) + ".png"


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route('/static/saved_preds/', methods=['GET'])
def download():
    nifti_img = mask_to_nifti(true_mask)
    return send_file(BytesIO(nifti_img.dataobj), as_attachment=True, download_name = "mask.bytes")

@app.route('/static/images/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


@app.route('/Project.html', methods=['GET',"POST"])
def index():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        extension = os.path.splitext(file.filename)[1]
        if extension not in app.config['ALLOWED_EXTENSIONS']:
            return render_template('Project.html', form = form, error = 'File is not a .nii file')
        #file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        #file_name = "D:\TSAProjects\static\\files\\"+file.filename
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)
        img = model_image(temp_file_path)
        os.remove(temp_file_path)
        os.rmdir(temp_dir)
        mask = mask_creation(img)
        name = display_mask(img,mask)
        file_url = url_for('get_file', filename = name)
        download = True
    else:
        file_url = None
        download = False
    return render_template('Project.html', form=form, file_url = file_url, download = download)



@app.route('/', methods=['GET',"POST"])    
@app.route('/index.html', methods=['GET',"POST"])
def home():
    return render_template('index.html')

@app.route('/contacts.html', methods=['GET',"POST"])
def contacts():
    return render_template('contacts.html')

@app.route('/Datasets.html', methods=['GET',"POST"])
def Datasets():
    return render_template('Datasets.html')

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port=5000)
