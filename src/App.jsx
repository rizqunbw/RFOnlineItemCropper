import React, { useState, useRef, useEffect } from 'react';
import ReactCrop from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import { Upload, Crop as CropIcon, Sparkles, Download, ScanSearch, X, Loader2 } from 'lucide-react';

function App() {
  const [upImg, setUpImg] = useState();
  const [crop, setCrop] = useState();
  const [completedCrop, setCompletedCrop] = useState(null);
  const [results, setResults] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeLog, setAnalyzeLog] = useState("Ready to analyze...");
  const imgRef = useRef(null);

  // Handle paste events globally
  useEffect(() => {
    const handlePaste = (e) => {
      if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
        // Trigger the file upload handler manually with the pasted file
        const file = e.clipboardData.files[0];
        const reader = new FileReader();
        reader.addEventListener('load', () => setUpImg(reader.result));
        reader.readAsDataURL(file);
        
        // Populate the hidden input logic for the FormData later
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        const fileInput = document.querySelector('.file-input');
        if(fileInput) fileInput.files = dataTransfer.files;
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => {
      window.removeEventListener('paste', handlePaste);
    };
  }, []);

  const onSelectFile = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const reader = new FileReader();
      reader.addEventListener('load', () => setUpImg(reader.result));
      reader.readAsDataURL(e.target.files[0]);
    }
  };

  const onLoad = (e) => {
    imgRef.current = e.currentTarget;
    setCrop(null); 
    setCompletedCrop(null);
    setAnalyzeLog("Image loaded. Click 'AI Auto-Detect' to find the item box.");
  };

  const autoDetectItemBox = async () => {
    if (!imgRef.current) return;
    setAnalyzing(true);
    setAnalyzeLog("Transmitting image to AI Python Backend...");
    
    try {
      const fileInput = document.querySelector('.file-input');
      if (!fileInput.files || fileInput.files.length === 0) {
        throw new Error("No file selected.");
      }
      
      const formData = new FormData();
      formData.append('image', fileInput.files[0]);
      
      setAnalyzeLog("Processing Canny Edge Detection Models...");
      
      const response = await fetch('/api/detect-box', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.success && data.box) {
        // The API now returns precise percentages
        const bestRect = data.box;
        
        // Pass the percentage directly to ReactCrop. 
        // This ensures the crop box stays glued to the item regardless of CSS responsive scaling.
        const newCrop = {
          unit: '%',
          x: Math.max(0, bestRect.x - 0.5), // slight padding
          y: Math.max(0, bestRect.y - 0.5),
          width: bestRect.width + 1,
          height: bestRect.height + 1
        };
        
        setCrop(newCrop);
        setCompletedCrop(newCrop);
        setAnalyzeLog(`AI detected item box.`);
      } else {
        setAnalyzeLog("Python API could not find the box. Applying fallback.");
        fallbackDetection();
      }
    } catch(err) {
      console.error(err);
      setAnalyzeLog("Error connecting to Python Backend. Make sure it is running on port 5000.");
      fallbackDetection();
    }
    setAnalyzing(false);
  };

  const fallbackDetection = () => {
    // Basic fallback: crop center 50%
    const img = imgRef.current;
    if(!img) return;
    const cw = img.width * 0.5;
    const ch = img.height * 0.5;
    const cx = (img.width - cw) / 2;
    const cy = (img.height - ch) / 2;
    const c = { unit: 'px', x: cx, y: cy, width: cw, height: ch };
    setCrop(c);
    setCompletedCrop(c);
    setAnalyzeLog("Heuristics: Cropped center region.");
    setAnalyzing(false);
  }

  const clearImage = () => {
    setUpImg(null);
    setCrop(null);
    setCompletedCrop(null);
    setResults([]);
    setAnalyzeLog("Ready to analyze...");
    if (imgRef.current) imgRef.current.src = "";
    const fileInput = document.querySelector('.file-input');
    if (fileInput) fileInput.value = "";
  };

  const generatePreview = () => {
    if (!completedCrop || !imgRef.current) return;
    const image = imgRef.current;
    const canvas = document.createElement('canvas');
    
    // completedCrop might be in % or px (DOM units). 
    // If it's px, we need to map DOM coordinate to natural image coordinate.
    let cropX, cropY, cropW, cropH;

    if (completedCrop.unit === '%') {
      cropX = (completedCrop.x / 100) * image.naturalWidth;
      cropY = (completedCrop.y / 100) * image.naturalHeight;
      cropW = (completedCrop.width / 100) * image.naturalWidth;
      cropH = (completedCrop.height / 100) * image.naturalHeight;
    } else {
      const scaleX = image.naturalWidth / image.width;
      const scaleY = image.naturalHeight / image.height;
      cropX = completedCrop.x * scaleX;
      cropY = completedCrop.y * scaleY;
      cropW = completedCrop.width * scaleX;
      cropH = completedCrop.height * scaleY;
    }

    // Ensure dimensions are valid
    if(cropW <= 0 || cropH <= 0) return;

    canvas.width = cropW;
    canvas.height = cropH;
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(
      image,
      cropX,
      cropY,
      cropW,
      cropH,
      0,
      0,
      cropW,
      cropH
    );

    const base64Image = canvas.toDataURL('image/png', 1.0);
    const newId = Date.now();
    
    setResults(prev => [{
      id: newId,
      url: base64Image,
      description: "AI Showcased Item",
      isReady: false
    }, ...prev]);

    // Give React time to render the image before allowing download (prevents race condition)
    setTimeout(() => {
      setResults(prev => prev.map(res => 
        res.id === newId ? { ...res, isReady: true } : res
      ));
    }, 1500);
  };



  return (
    <div className="app-container">
      <div className="header">
        <h1 className="title">RF Online <span>Item Cropper</span></h1>
        <p className="subtitle">AI-powered tool to crop and extract high-quality details of in-game items from RF Online screenshots.</p>
      </div>

      <div className="workspace">
        <div className="panel">
          <div className="controls">
            <input type="file" accept="image/*" onChange={onSelectFile} className="file-input" id="hidden-upload" />
            
            {upImg && (
              <>
                 <button className="btn" onClick={clearImage}>
                   <X size={18} /> Close Image
                 </button>
                 <button className="btn magic" onClick={autoDetectItemBox} disabled={analyzing}>
                   <Sparkles size={18} />
                   {analyzing ? 'Scanning Target...' : 'AI Auto-Detect Box'}
                 </button>
              </>
            )}

            {completedCrop && (
               <button className="btn primary btn-attention" onClick={generatePreview}>
                 <CropIcon size={18} />
                 Extract Showcased Item
               </button>
            )}
          </div>

          <div className="canvas-container">
            {upImg ? (
              <ReactCrop
                crop={crop}
                onChange={(_, percentCrop) => setCrop(percentCrop)}
                onComplete={(c) => setCompletedCrop(c)}
              >
                <img ref={imgRef} src={upImg} onLoad={onLoad} alt="Upload" />
              </ReactCrop>
            ) : (
              <div className="empty-state">
                <ScanSearch className="empty-icon" size={64} />
                <p style={{fontSize: '1.2rem', fontWeight: 600}}>Ready for scanning...</p>
                <p>Press <strong>CTRL+V</strong> to paste a screenshot directly or</p>
                <label className="btn primary" htmlFor="hidden-upload" style={{marginTop: '0.8rem'}}>
                  <Upload size={18} /> Upload Image
                </label>
              </div>
            )}
          </div>
          
          {upImg && (
            <div className="analysis-text">
              {analyzeLog}
            </div>
          )}
        </div>

        {results.length > 0 && (
          <div className="panel" style={{flex: 0.5}}>
             <h3 style={{color: 'white', marginBottom: '1rem', fontWeight: 600, fontSize: '1.2rem'}}>Output Showcase</h3>
             <div className="results-grid">
               {results.map((res) => (
                    <div className="result-card" key={res.id}>
                      <img src={res.url} alt="Showcase result" />
                      <div><span className="badge">{res.description}</span></div>
                      {res.isReady ? (
                        <a 
                          className="btn" 
                          style={{marginTop:'1.2rem', width: '100%', fontSize: '0.9rem', justifyContent: 'center', textDecoration: 'none', display: 'flex', gap: '0.5rem', alignItems: 'center'}} 
                          href={res.url}
                          download={`rf-showcase-${res.id}.png`}
                        >
                          <Download size={16} /> Save Extraction
                        </a>
                      ) : (
                        <button 
                          className="btn" 
                          disabled
                          style={{marginTop:'1.2rem', width: '100%', fontSize: '0.9rem', justifyContent: 'center', opacity: 0.7, cursor: 'not-allowed'}} 
                        >
                          <Loader2 size={16} className="spinner" /> Preparing PNG...
                        </button>
                      )}
                    </div>
               ))}
             </div>
          </div>
        )}
      </div>

      <footer className="footer">
        Crafted by <a href="https://www.facebook.com/rizqubocinista" target="_blank" rel="noreferrer">Bocinista</a>
      </footer>
    </div>
  );
}

export default App;
