const chooseImageButton = document.querySelector('#chooseImageButton');
const uploadInput = document.querySelector('#upload');
const convertButton = document.querySelector('#convertButton');
const submitInput = document.querySelector('#submit');
const imagePathInput = document.querySelector('#imagePathInput');
const imagePreview = document.querySelector('#imagePreview');
const typewriter = document.querySelector('#typewriter');
const boxContainer = document.querySelector('.box-container');


chooseImageButton.addEventListener('click', () => {
    uploadInput.click();
});

uploadInput.addEventListener('change', () => {
    const selectedFile = uploadInput.files[0];
    if (selectedFile) {
        imagePathInput.value = selectedFile.name;
        const reader = new FileReader();

        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.classList.add('selected-image');
        };

        reader.readAsDataURL(selectedFile);
    } else {
        imagePathInput.value = '';
    }

});


convertButton.addEventListener('click', async function(e) {
    e.preventDefault();
    submitInput.click();
    const selectedFile = uploadInput.files[0];
    let response = await getCode(selectedFile);
    let text = response['code']
    console.log(text)
    type(text, 0)
});


async function getCode(selectedFile){
    let response = "";
    if (selectedFile) {
        const formData = new FormData();
        formData.append('img', selectedFile);
        response = await fetch('http://localhost:8080/predict', {
                                        method: 'POST',
                                        body: formData
                                    });
        response = response.json()
    }
    else{
        alert("Please select an image")
    }
    return response;
}


function type(text, index) {
    if (index < text.length) {
        const typedText = text.slice(0, index).replace(/\n/g, '<br>');
        typewriter.innerHTML = typedText + '<span class="blinking-cursor">|</span>';
        index++;
        setTimeout(type, 5, text, index);
    } else {
        const typedText = text.slice(0, index).replace(/\n/g, '<br>');
        typewriter.innerHTML = typedText;
    }

    if (typewriter.textContent.trim() !== '') {
        boxContainer.classList.remove('hidden');
    }
}