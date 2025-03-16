const { spawn } = require('child_process');

function cleanData(data, callback) {
    const python = spawn('python', ['utils/dataCleaner.py']);

    let cleanedData = "";

    python.stdout.on('data', (data) => {
        cleanedData += data.toString();
    });

    python.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });

    python.on('close', (code) => {
        if (code === 0) {
            try {
                callback(null, JSON.parse(cleanedData));
            } catch (err) {
                callback(`JSON Parsing Error: ${err.message}`, null);
            }
        } else {
            callback(`Python process exited with code ${code}`, null);
        }
    });

    python.stdin.write(JSON.stringify(data));
    python.stdin.end();
}

module.exports = cleanData;
