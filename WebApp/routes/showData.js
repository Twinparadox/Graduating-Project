var express = require('express');
var router = express.Router();
var mysql_dbc = require('../config/db_con')();
var connection = mysql_dbc.init();

router.get('/', function(req, res, next) {
    var days = req.query.days;
    var corp = req.query.corp;
    console.log('[showData] GET Request');

    var sql = "SELECT trading_price FROM table_logs WHERE trading_corp='"+corp+"' ORDER BY trading_id DESC LIMIT " + days;
    connection.query(sql, function(error, rows, fields) {
        if(!error) {
            for(var i = 0;i < rows.length;i++) {
                console.log(rows[i]);
            }
            res.render('show-all-data', {results:rows});
        } else {
            console.log("query error : "+error);
        }
    })
})

router.post('/', function(req, res, next) {
    var days = req.body.days;
    var corp = req.body.corp;
    console.log('[showData] POST Request');
    console.log(req.body)
    var sql = "SELECT * FROM table_logs WHERE trading_corp='"+corp+"' ORDER BY trading_id DESC LIMIT " + days;
    connection.query(sql, function(error, rows, fields) {
        if(!error) {
            var result = {corp:corp, trading_date:[], trading_price:[], trading_volume:[],
                          trading_action:[], trading_profit:[], trading_prefix:[]};
            
            // JSON 형식을 TOAST CHART에서 쓰기 쉽게 변환
            // 성능에 문제 있으면 변경 고려
            for(var i = rows.length - 1;i >= 0;i--) {
                result.trading_date.push(rows[i].trading_date.toISOString().slice(0,10));
                result.trading_price.push(rows[i].trading_price);
                result.trading_volume.push(rows[i].trading_volume);
                result.trading_action.push(rows[i].trading_action);
                result.trading_profit.push(rows[i].trading_profit);
                result.trading_prefix.push(rows[i].trading_prefix);
            }
            console.log(rows);
            res.json(result);
        } else {
            console.log("query error : "+error);
        }
    })
})

module.exports = router;