const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const dataRoute = require('./routes/dataRoute');  

const app = express();


app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));


app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());



app.use('/data', dataRoute);  


app.get('/', (req, res) => {
    res.render('index');
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
