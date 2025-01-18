console.log(faceapi)

const run = async()=>{
    let stream= await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    })

    let videoFeed=document.querySelector(".video-feed");
    videoFeed.srcObject=stream;

    // Pre Trained ML Models
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
        faceapi.nets.ageGenderNet.loadFromUri('./models'),
        faceapi.nets.faceExpressionNet.loadFromUri('./models')
    ])

    // Make the canvas of same size and postion as of video feed
    let canvas=document.querySelector("#canvas");
    canvas.style.left=videoFeed.offsetLeft;
    canvas.style.top=videoFeed.offsetTop;
    canvas.height=videoFeed.height;
    canvas.width=videoFeed.width;

    // Face Recognition with Michael Jordan
    let refFace=await faceapi.fetchImage("https://imgs.search.brave.com/kH2AJqx2IkvxEaatTzUc6OEEgLT617OW01rarxfI7RY/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZXR0eWltYWdl/cy5jb20vaWQvNTE1/NDYxNzcwL3Bob3Rv/L21pY2hhZWwtam9y/ZGFuLWZvcndhcmQt/Zm9yLXRoZS1jaGlj/YWdvLWJ1bGxzLWlu/LXRoZS1sb2NrZXIt/cm9vbS5qcGc_cz02/MTJ4NjEyJnc9MCZr/PTIwJmM9YnBIbVR0/TzdtbGlKT1JsWjlx/MzI1X1lIVGZOZWNB/aWpFUHNTN1hQenlw/MD0");

    let refFaceAiData=await faceapi.detectAllFaces(refFace).withFaceLandmarks().withFaceDescriptors();

    let faceMatcher=new faceapi.FaceMatcher(refFaceAiData);

    // Facial Detection after every each 200 milliseconds
    setInterval(async ()=>{
        // We have real time facial detection data in detections variable
        let detections=await faceapi.detectAllFaces(videoFeed).withFaceLandmarks().withFaceDescriptors().withAgeAndGender().withFaceExpressions();

        // Clearing the canvas
        canvas.getContext("2d").clearRect(0,0,canvas.width,canvas.height);
        detections=faceapi.resizeResults(detections,videoFeed);

        // Drawing detection, landmarks, expression over canvas
        faceapi.draw.drawDetections(canvas,detections);
        faceapi.draw.drawFaceLandmarks(canvas,detections);
        faceapi.draw.drawFaceExpressions(canvas,detections);

        // Drawing Gender and Age over canvas
        detections.forEach((face)=>{
            let {age, gender, genderProbability,detection,descriptor}=face;
            let ageText=`${Math.round(age)} years`;
            let textField=new faceapi.draw.DrawTextField([gender,ageText],face.detection.box.topRight)
            textField.draw(canvas);

            let label=faceMatcher.findBestMatch(descriptor).toString();
            
            // Create label options
            let options = {label: label.includes("unknown") ? "Unknown Subject..." : "Michael Jordan" };

            // Create the box with the label and draw it
            let drawBox = new faceapi.draw.DrawBox(detection.box, options);
            drawBox.draw(canvas);
        })
    },200)

    
}

run()
