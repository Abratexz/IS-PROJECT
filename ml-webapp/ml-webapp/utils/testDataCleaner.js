const cleanData = require('./dataCleaner');  // Import dataCleaner.js

const sampleData = [
    { "name": "Alice", "age": 25 },
    { "name": "Bob", "age": 30 },
    { "name": "Alice", "age": 25 } // Duplicate row
];

cleanData(sampleData, (err, cleaned) => {
    if (err) {
        console.error("Error:", err);
    } else {
        console.log("Cleaned Data:", cleaned);
    }
});
