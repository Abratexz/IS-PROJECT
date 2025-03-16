// routes/dataRoute.js

const express = require('express');
const router = express.Router();
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const cleanData = require('../utils/dataCleaner');
const { spawn } = require('child_process');

// ------------------------------------------
// ช่วยฟังก์ชันใช้งาน
// ------------------------------------------

// ระบุ path Python สำหรับ Render (หรือระบบอื่นๆ)
const pythonExecutable = '/usr/bin/python3'; 

function collectColumns(dataArray) {
  const colSet = new Set();
  dataArray.forEach((row) => {
    Object.keys(row).forEach((col) => colSet.add(col));
  });
  return Array.from(colSet);
}

function readCSVFile(filepath, callback) {
  const results = [];
  fs.createReadStream(filepath)
    .pipe(csv())
    .on('data', (data) => results.push(data))
    .on('end', () => callback(null, results))
    .on('error', (err) => callback(err, null));
}

// ------------------------------------------
// ตัวอย่างเส้นทางพื้นฐาน
// ------------------------------------------

router.get('/', (req, res) => {
  res.render('index');
});

// ------------------------------------------
// Preview Raw CSV
// ------------------------------------------

router.get('/raw-mobile-data', (req, res) => {
  const csvPath = path.join(__dirname, '../data/mobile_dataset.csv');
  readCSVFile(csvPath, (err, data) => {
    if (err) {
      return res.status(500).send(`Error reading mobile dataset: ${err}`);
    }
    const columns = collectColumns(data);
    res.render('rawDataPreview', {
      title: 'Raw Mobile Dataset',
      columns,
      tableData: data
    });
  });
});

router.get('/raw-worker-data', (req, res) => {
  const csvPath = path.join(__dirname, '../data/worker_dataset.csv');
  readCSVFile(csvPath, (err, data) => {
    if (err) {
      return res.status(500).send(`Error reading worker dataset: ${err}`);
    }
    const columns = collectColumns(data);
    res.render('rawDataPreview', {
      title: 'Raw Worker Dataset',
      columns,
      tableData: data
    });
  });
});

router.get('/raw-cnn-data', (req, res) => {
  const csvPath = path.join(__dirname, '../data/cnn_dataset.csv');
  readCSVFile(csvPath, (err, data) => {
    if (err) {
      return res.status(500).send(`Error reading cnn dataset: ${err}`);
    }
    // ตัวอย่าง: แสดงแค่ 100 แถวแรก
    const slicedData = data.slice(0, 100);
    const columns = collectColumns(slicedData);
    res.render('rawDataPreview', {
      title: 'Raw CNN PIXEL Dataset',
      columns,
      tableData: slicedData
    });
  });
});

// ------------------------------------------
// Clean CSV
// ------------------------------------------

router.get('/clean-mobile-data', (req, res) => {
  const filepath = path.join(__dirname, '../data/mobile_dataset.csv');
  readCSVFile(filepath, (err, jsonData) => {
    if (err) return res.status(500).json({ error: err });
    cleanData(jsonData, (err, cleaned) => {
      if (err) return res.status(500).json({ error: err });
      res.render('cleanedDataPreview', {
        columns: Object.keys(cleaned[0]),
        data: cleaned,
        datasetType: 'mobile'
      });
    });
  });
});

router.get('/clean-worker-data', (req, res) => {
  const filepath = path.join(__dirname, '../data/worker_dataset.csv');
  readCSVFile(filepath, (err, jsonData) => {
    if (err) return res.status(500).json({ error: err });
    cleanData(jsonData, (err, cleaned) => {
      if (err) return res.status(500).json({ error: err });
      res.render('cleanedDataPreview', {
        columns: Object.keys(cleaned[0]),
        data: cleaned,
        datasetType: 'worker'
      });
    });
  });
});

// ------------------------------------------
// Analyze CSV with dataAnalysis.py
// ------------------------------------------

router.get('/analyze-mobile', (req, res) => {
  const filePath = path.join(__dirname, '../data/mobile_dataset.csv');
  runAnalysis(filePath, res);
});

router.get('/analyze-worker', (req, res) => {
  const filePath = path.join(__dirname, '../data/worker_dataset.csv');
  runAnalysis(filePath, res);
});

function runAnalysis(filePath, res) {
  readCSVFile(filePath, (err, jsonData) => {
    if (err) {
      return res.status(500).send(`Error reading CSV: ${err}`);
    }
    const scriptPath = path.join(__dirname, '../utils/dataAnalysis.py');
    const pyProcess = spawn(pythonExecutable, [scriptPath]);

    let pyData = "";
    pyProcess.stdout.on('data', (data) => {
      pyData += data.toString();
    });

    pyProcess.stderr.on('data', (data) => {
      console.error(`Python Error: ${data}`);
    });

    pyProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(pyData);
          res.render('analysisPreview', {
            summary: result.summary_stats,
            plotFilename: result.plot_filename
          });
        } catch (err) {
          res.status(500).send(`Failed to parse Python output: ${err}`);
        }
      } else {
        res.status(500).send(`Python process exited with code ${code}`);
      }
    });

    // ส่งข้อมูล CSV ให้ Python script ทาง stdin
    pyProcess.stdin.write(JSON.stringify(jsonData));
    pyProcess.stdin.end();
  });
}

// ------------------------------------------
// Train Model (Linear Regression) for mobile
// ------------------------------------------

router.get('/train-mobile-with-clean', (req, res) => {
  const csvPath = path.join(__dirname, '../data/mobile_dataset.csv');
  readCSVFile(csvPath, (err, rawData) => {
    if (err) {
      return res.render('trainResult', {
        error: `Error reading CSV: ${err}`,
        message: null, modelPath: null, MSE: null, R2: null
      });
    }

    // เรียก dataCleaner.py
    const cleanerScript = path.join(__dirname, '../utils/dataCleaner.py');
    let cleanedOutput = '';
    const cleaner = spawn(pythonExecutable, [cleanerScript]);

    cleaner.stdout.on('data', (data) => {
      cleanedOutput += data.toString();
    });

    cleaner.stderr.on('data', (data) => {
      console.error('Cleaner Python Error:', data.toString());
    });

    cleaner.on('close', (code) => {
      if (code !== 0) {
        return res.render('trainResult', {
          error: `Cleaner script exited with code ${code}`,
          message: null, modelPath: null, MSE: null, R2: null
        });
      }
      let cleanedData;
      try {
        cleanedData = JSON.parse(cleanedOutput);
      } catch (e) {
        return res.render('trainResult', {
          error: `Failed to parse cleaned JSON: ${e}`,
          message: null, modelPath: null, MSE: null, R2: null
        });
      }

      // เรียก trainModel.py
      const trainerScript = path.join(__dirname, '../utils/trainModel.py');
      let trainOutput = '';
      const trainer = spawn(pythonExecutable, [trainerScript]);

      trainer.stdout.on('data', (data) => {
        trainOutput += data.toString();
      });

      trainer.stderr.on('data', (data) => {
        console.error('Trainer Python Error:', data.toString());
      });

      trainer.on('close', (code) => {
        if (code !== 0) {
          return res.render('trainResult', {
            error: `Trainer script exited with code ${code}`,
            message: null, modelPath: null, MSE: null, R2: null
          });
        }
        try {
          const trainResult = JSON.parse(trainOutput);
          res.render('trainResult', {
            error: null,
            message: trainResult.message,
            modelPath: trainResult.model_path,
            MSE: trainResult.MSE,
            R2: trainResult.R2,
            MAE: trainResult.MAE,
            MedianAE: trainResult.MedianAE,
            Coefficients: trainResult.Coefficients,
            Intercept: trainResult.Intercept
          });
        } catch (err) {
          res.render('trainResult', {
            error: `Failed to parse train output: ${err}`,
            message: null, modelPath: null, MSE: null, R2: null
          });
        }
      });
      // ส่งข้อมูลที่ผ่าน cleaner แล้วเข้า trainer
      trainer.stdin.write(JSON.stringify(cleanedData));
      trainer.stdin.end();
    });
    // ส่ง rawData เข้า cleaner
    cleaner.stdin.write(JSON.stringify(rawData));
    cleaner.stdin.end();
  });
});

// ------------------------------------------
// Predict with model (mobile)
// ------------------------------------------

router.post('/predict-mobile', (req, res) => {
  const scriptPath = path.join(__dirname, '../utils/predictModel.py');
  const pyProcess = spawn(pythonExecutable, [scriptPath]);
  let outputData = '';

  pyProcess.stdout.on('data', (data) => {
    outputData += data.toString();
  });

  pyProcess.stderr.on('data', (data) => {
    console.error('[predict-mobile Error]:', data.toString());
  });

  pyProcess.on('close', (code) => {
    if (code === 0) {
      try {
        const result = JSON.parse(outputData);
        const formattedPrediction = parseFloat(result.prediction).toFixed(2);
        res.render('index', {
          mobilePrediction: formattedPrediction,
          workerPrediction: null,
          error: null
        });
      } catch (err) {
        res.render('index', {
          mobilePrediction: null,
          workerPrediction: null,
          error: `Parsing error: ${err}`
        });
      }
    } else {
      res.render('index', {
        mobilePrediction: null,
        workerPrediction: null,
        error: `Python exited with code ${code}`
      });
    }
  });

  // ส่ง req.body เข้าไปใน stdin ของ Python
  pyProcess.stdin.write(JSON.stringify(req.body));
  pyProcess.stdin.end();
});

// ------------------------------------------
// Train Decision Tree for Worker
// ------------------------------------------

router.get('/train-mobile-dt', (req, res) => {
  const filePath = path.join(__dirname, '../data/worker_dataset.csv');
  readCSVFile(filePath, (err, jsonData) => {
    if (err) {
      return res.render('trainResult', {
        error: `Error reading CSV: ${err}`,
        message: null,
        modelPath: null,
        MAE: null,
        MedianAE: null,
        MSE: null,
        R2: null,
        Coefficients: null,
        Intercept: null,
      });
    }
    const scriptPath = path.join(__dirname, '../utils/trainDecisionTree.py');
    const pyProcess = spawn(pythonExecutable, [scriptPath]);
    let outputData = '';

    pyProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });

    pyProcess.stderr.on('data', (data) => {
      console.error('Python Error:', data.toString());
    });

    pyProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(outputData);
          res.render('trainResult', {
            error: null,
            message: result.message,
            modelPath: result.model_path,
            MAE: result.MAE,
            MedianAE: result.MedianAE,
            MSE: result.MSE,
            R2: result.R2,
            Coefficients: result.Coefficients,
            Intercept: result.Intercept
          });
        } catch (err) {
          res.render('trainResult', {
            error: `Could not parse Python output: ${err}`,
            message: null,
            modelPath: null,
            MAE: null,
            MedianAE: null,
            MSE: null,
            R2: null,
            Coefficients: null,
            Intercept: null,
          });
        }
      } else {
        res.render('trainResult', {
          error: `Python script exited with code ${code}`,
          message: null,
          modelPath: null,
          MAE: null,
          MedianAE: null,
          MSE: null,
          R2: null,
          Coefficients: null,
          Intercept: null,
        });
      }
    });

    // ส่งข้อมูลเข้า python
    pyProcess.stdin.write(JSON.stringify(jsonData));
    pyProcess.stdin.end();
  });
});

// ------------------------------------------
// Predict Worker
// ------------------------------------------

router.post('/predict-worker', (req, res) => {
  const scriptPath = path.join(__dirname, '../utils/predictWorker.py');
  const pyProcess = spawn(pythonExecutable, [scriptPath]);
  let outputData = '';

  pyProcess.stdout.on('data', (data) => {
    outputData += data.toString();
  });

  pyProcess.stderr.on('data', (data) => {
    console.error('[predict-worker Error]:', data.toString());
  });

  pyProcess.on('close', (code) => {
    if (code === 0) {
      try {
        const result = JSON.parse(outputData);
        const formattedPrediction = parseFloat(result.prediction).toFixed(2);
        res.render('index', {
          mobilePrediction: null,
          workerPrediction: formattedPrediction,
          error: null
        });
      } catch (err) {
        res.render('index', {
          mobilePrediction: null,
          workerPrediction: null,
          error: `Parsing error: ${err}`
        });
      }
    } else {
      res.render('index', {
        mobilePrediction: null,
        workerPrediction: null,
        error: `Python exited with code ${code}`
      });
    }
  });

  // ส่งข้อมูล form ให้ python
  pyProcess.stdin.write(JSON.stringify(req.body));
  pyProcess.stdin.end();
});

// ------------------------------------------
// Train & Predict CNN (Numbers) 
// ------------------------------------------

router.post("/train-number-model", (req, res) => {
  const scriptPath = path.join(__dirname, '../utils/trainNumberModel.py');
  const pyProcess = spawn(pythonExecutable, [scriptPath]);

  let resultData = "";
  pyProcess.stdout.on("data", (data) => {
    resultData += data.toString();
  });

  pyProcess.stderr.on("data", (data) => {
    console.error("[trainNumberModel.py Error]:", data.toString());
  });

  pyProcess.on("close", (code) => {
    if (code === 0) {
      try {
        const output = JSON.parse(resultData);
        res.render("TrainResultCNN", {
          message: output.message,
          modelPath: output.model_path,
          finalLoss: output.final_loss,
          epochs: output.epochs
        });
      } catch (error) {
        res.render("TrainResultCNN", {
          message: "Training failed. Check logs.",
          modelPath: null,
          finalLoss: null,
          epochs: null
        });
      }
    } else {
      res.render("TrainResultCNN", {
        message: `Python script exited with code ${code}`,
        modelPath: null,
        finalLoss: null,
        epochs: null
      });
    }
  });
});

router.post("/predict-number", (req, res) => {
  const { number } = req.body;
  const scriptPath = path.join(__dirname, '../utils/predictNumber.py');
  const pyProcess = spawn(pythonExecutable, [scriptPath, number]);

  let resultData = "";
  pyProcess.stdout.on("data", (data) => {
    resultData += data.toString();
  });

  pyProcess.stderr.on("data", (data) => {
    console.error("[predictNumber.py Error]:", data.toString());
  });

  pyProcess.on("close", (code) => {
    if (code === 0) {
      try {
        const output = JSON.parse(resultData);
        res.render("index", {
          singleNumberPrediction: output.predicted_class
        });
      } catch (error) {
        res.render("index", {
          singleNumberPrediction: "Error: Could not predict!"
        });
      }
    } else {
      res.render("index", {
        singleNumberPrediction: `Python exited with code ${code}`
      });
    }
  });
});

// ------------------------------------------
// Explanation Page
// ------------------------------------------
router.get("/explanation-algorithm-theory", (req, res) => {
  res.render("algorithm-theory");
});

// ------------------------------------------
// Exports
// ------------------------------------------
module.exports = router;
