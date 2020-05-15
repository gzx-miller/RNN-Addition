import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

// 字符表'0123456789+ '
class CharacterTable {
    constructor(chars) {
        this.chars = chars;
        this.charIndices = {};
        this.indicesChar = {};
        this.size = this.chars.length;                                // 长度
        for (let i = 0; i < this.size; ++i) {
            const char = this.chars[i];
            if (this.charIndices[char] != null) {                     // 遇到重复就抛异常
                throw new Error(`Duplicate character '${char}'`);
            }
            this.charIndices[this.chars[i]] = i;                      // 索引: 字符->序号
            this.indicesChar[i] = this.chars[i];                      // 索引: 序号->字符
        }
    }

    // 将算式编码成张量
    encode(str, numRows) {                                            // 字符串, 串的最大长度
        const buf = tf.buffer([numRows, this.size]);                  // 创建行列张量
        for (let i = 0; i < str.length; ++i) {
            const char = str[i];
            if (this.charIndices[char] == null) {                      // 不存在的字符, 抛出异常
                throw new Error(`Unknown character: '${char}'`);
            }
            buf.set(1, i, this.charIndices[char]);                     // 字符表中对应的位置设置 1
        }
        return buf.toTensor().as2D(numRows, this.size);                // 转成二维
    }

    // 将多组算式编码成张量
    encodeBatch(strings, numRows) {                                    // 字符串, 串的最大长度
        const numExamples = strings.length;
        const buf = tf.buffer([numExamples, numRows, this.size]);      // 创建行列张量
        for (let n = 0; n < numExamples; ++n) {
            const str = strings[n];
            for (let i = 0; i < str.length; ++i) {
                const char = str[i];
                if (this.charIndices[char] == null) {                  // 不存在的字符, 抛出异常
                    throw new Error(`Unknown character: '${char}'`);
                }
                buf.set(1, n, i, this.charIndices[char]);              // 字符表中对应的位置设置 1
            }
        }
        return buf.toTensor().as3D(numExamples, numRows, this.size);   // 转成三维
    }

    // 将张量转回算式
    decode(x, calcArgmax = true) {
        return tf.tidy(() => {                      // 自动对内存回收
            if (calcArgmax) {
                x = x.argMax(1);                    // 每行的最大值索引
            }
            const xData = x.dataSync();             // 同步从张量取值, 性能不佳, 异步使用data()替换
            let output = '';
            for (const index of Array.from(xData)) {
                output += this.indicesChar[index];
            }
            return output;
        });
    }
}

// 产生目标长度的随机十进制数字
const digitArray = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];    // 字符 0~9
const arraySize = digitArray.length;                               // 字符个数
const randomInt = (digits) => {
    let str = '';
    while (str.length < digits) {
        const index = Math.floor(Math.random() * arraySize);       // 每位数字都随机
        str += digitArray[index];
    }
    return Number.parseInt(str);
};

// 将多组算式+答案转换成张量
function convertDataToTensors(data, charTable, digits) {
    const questions = data.map(item => item[0]);             // [算式]
    const answers = data.map(item => item[1]);               // [答案]
    return [
        charTable.encodeBatch(questions, digits * 2 + 1),    // 算式编码成张量
        charTable.encodeBatch(answers, digits + 1),          // 答案编码成张量
    ];
}

// 创建训练模型
function createAndCompileModel(
    layers, hiddenSize, rnnType, digits, vocabularySize) {
    const maxLen = 2 * digits + 1;

    const model = tf.sequential();                       // 创建多层网模型
    function addRNN(retSeq) {
        let opt = {
            units: hiddenSize,                            // 隐藏层单元数
            recurrentInitializer: 'glorotNormal',         // 回归初始化
        }
        if (retSeq) opt.returnSequences = true;
        else opt.inputShape = [maxLen, vocabularySize];

        switch (rnnType) {                                  // RNN 类型
            case 'SimpleRNN':                               // 简单RNN
                model.add(tf.layers.simpleRNN(opt));
                break;
            case 'GRU':                                     // 门控循环单元网络
                model.add(tf.layers.gru(opt));
                break;
            case 'LSTM':                                    // 长短期记忆网络
                model.add(tf.layers.lstm(opt));
                break;
            default:
                throw new Error(`Unsupported RNN type: '${rnnType}'`);
        }
    }
    addRNN(false);                                           // 添加一个RNN
    model.add(tf.layers.repeatVector({ n: digits + 1 }));    // 添加一个重复层, 重复n次
    addRNN(true);                                            // 添加一个RNN, 有返回
    model.add(tf.layers.timeDistributed(                     // 添加全连接层 
        { layer: tf.layers.dense({ units: vocabularySize }) }));
    model.add(tf.layers.activation({ activation: 'softmax' }));      // 添加softmax激活层
    model.compile({
        loss: 'categoricalCrossentropy',                     // 损失函数
        optimizer: 'adam',                                   // 优化器
        metrics: ['accuracy']                                // 精确: 不同的矩阵对应不同的输出
    });
    return model;
}

// 生成算式和答案
function generateData(digits, numExamples) {               // 数字位数, 个数
    const output = [];
    const maxLen = digits * 2 + 1;                         // 算式长度

    const seen = new Set();                                // 去重
    while (output.length < numExamples) {                  // 不超出个数
        const a = randomInt(digits);
        const b = randomInt(digits);
        const sorted = b > a ? [a, b] : [b, a];
        let quest = `${sorted[0]}+${sorted[1]}`;            // 算式
        if (seen.has(quest)) continue;                      // 重复就丢弃
        seen.add(quest);

        quest += ' '.repeat(maxLen - quest.length);         // 空格补齐算式长度 2*digits+1
        let ans = (a + b).toString();                       // 答案
        ans += ' '.repeat(digits + 1 - ans.length);         // 空格补齐答案长度 digits+1
        output.push([quest, ans]);                          // 输出[[算式, 答案]...]
    }
    return output;
}

// 训练模型对象
class AdditionRNN {
    constructor(digits, trainingSize, rnnType, layers, hiddenSize) {
        this.digits = digits;
        this.trainingSize = trainingSize;
        this.rnnType = rnnType;
        this.layers = layers;
        this.hiddenSize = hiddenSize;
        this.trainXs = this.trainYs = null;
        this.testXs = this.testYs = null;
        this.textXsForDisplay = null;
        var chars = '0123456789+ ';                                 // 字符表 0~9
        this.charTable = new CharacterTable(chars);
        this.model = createAndCompileModel(                         // 创建模型
            this.layers, this.hiddenSize, this.rnnType, this.digits, chars.length);
    }
    initDatas() {
        console.log('Generating training data');
        const data = generateData(this.digits, this.trainingSize);            // 生成算式和答案
        const split = Math.floor(this.trainingSize * 0.9);
        this.trainData = data.slice(0, split);                      // 九成用于训练
        this.testData = data.slice(split);                          // 一成用于测验
        [this.trainXs, this.trainYs] =                              // 转成张量(算式, 答案)
            convertDataToTensors(this.trainData, this.charTable, this.digits);
        [this.testXs, this.testYs] =
            convertDataToTensors(this.testData, this.charTable, this.digits);

    }
    // 训练
    async train(iterations, batchSize, numTestExamples) {
        const lossValues = [[], []];                                  // 损失
        const accuracyValues = [[], []];                              // 精确度
        for (let i = 0; i < iterations; ++i) {                        // 循环次数
            this.initDatas();
            const beginMs = performance.now();
            const history = await this.model.fit(                     // 将算式和答案放入训练
                this.trainXs, this.trainYs, {
                    epochs: 1,                                        // 周期数
                    batchSize,                                        // 一批数据的大小
                    validationData: [this.testXs, this.testYs],       // 校验数据
                    yieldEvery: 'epoch'                               // 每周期可终止
                });

            const modelFitTime = (performance.now() - beginMs) / 1000;     // 消耗时长
            const trainLoss = history.history['loss'][0];                  // 训练损失
            const trainAccuracy = history.history['acc'][0];               // 训练准确度
            const valLoss = history.history['val_loss'][0];                // 校验损失
            const valAccuracy = history.history['val_acc'][0];             // 校验准确度

            document.getElementById('trainStatus').textContent =           // 描述当前迭代序号和时长
                `Iteration ${i + 1} of ${iterations}: Duration: ${modelFitTime.toFixed(3)} (s)`;

            const lossContainer = document.getElementById('lossChart');         // 损失填充到图表
            lossValues[0].push({ 'x': i, 'y': trainLoss });
            lossValues[1].push({ 'x': i, 'y': valLoss });
            tfvis.render.linechart(lossContainer,
                { values: lossValues, series: ['train', 'validation'] },
                { width: 420, height: 300, xLabel: 'epoch', yLabel: 'loss' }
            );

            const accuracyContainer = document.getElementById('accuracyChart');  // 精确度填充到图表
            accuracyValues[0].push({ 'x': i, 'y': trainAccuracy });
            accuracyValues[1].push({ 'x': i, 'y': valAccuracy });
            tfvis.render.linechart(accuracyContainer,
                { values: accuracyValues, series: ['train', 'validation'] },
                { width: 420, height: 300, xLabel: 'epoch', yLabel: 'accuracy' }
            );

            if (this.textXsForDisplay) this.textXsForDisplay.dispose();
            this.testXsForDisplay = this.testXs.slice([0, 0, 0],                 // 用于预测
                [numTestExamples, this.testXs.shape[1], this.testXs.shape[2]]
            );

            const examples = [];
            const isCorrect = [];
            tf.tidy(() => {                                                      // 包含内存回收
                const predictOut = this.model.predict(this.testXsForDisplay);    // 预测, 返回预测结果
                for (let k = 0; k < numTestExamples; ++k) {                      // 遍历预测结果
                    const scores = predictOut.slice([k, 0, 0],                   // 第K个结果
                        [1, predictOut.shape[1], predictOut.shape[2]]
                    ).as2D(predictOut.shape[1], predictOut.shape[2]);
                    const decoded = this.charTable.decode(scores);               // 解码成字符
                    examples.push(this.testData[k][0] + ' = ' + decoded);        // 放到输出对象
                    isCorrect.push(this.testData[k][1].trim() === decoded.trim());  // 对错
                }
            });

            const examplesDiv = document.getElementById('testExamples');          // 展示预测结果
            const examplesContent = examples.map((example, i) =>                  // 样式: 错红/对绿
                `<div class="${isCorrect[i] ? 'answer-correct' : 'answer-wrong'}">${example}</div>`
            );
            examplesDiv.innerHTML = examplesContent.join('\n');
        }
    }
}

async function runAdditionRNNDemo() {
    document.getElementById('trainModel').addEventListener('click', async () => {
        const digits = +(document.getElementById('digits')).value;                // 数字位数
        const trainingSize = +(document.getElementById('trainingSize')).value;    // 训练量
        const select = document.getElementById('rnnType');
        const rnnType =
            select.options[select.selectedIndex].getAttribute('value');           // RNN类型
        const layers = +(document.getElementById('rnnLayers')).value;             // 神经网层数
        const hiddenSize = +(document.getElementById('rnnLayerSize')).value;      // 隐藏层大小
        const batchSize = +(document.getElementById('batchSize')).value;          // 学习量
        const trainIterations = +(document.getElementById('trainIterations')).value;     // 训练迭代次数
        const numTestExamples = +(document.getElementById('numTestExamples')).value;     // 展示的测试个数

        const status = document.getElementById('trainStatus');                    // 训练的状态
        if (digits < 1 || digits > 5) {                                           // 数字长度不大于5
            status.textContent = 'digits must be >= 1 and <= 5';
            return;
        }
        const trainingSizeLimit = Math.pow(Math.pow(10, digits), 2);              // 限制训练数量
        if (trainingSize > trainingSizeLimit) {
            status.textContent =
                `With digits = ${digits}, you cannot have more than ` +
                `${trainingSizeLimit} examples`;
            return;
        }

        const demo = new AdditionRNN(                                             // 创建训练模型
            digits, trainingSize, rnnType, layers, hiddenSize
        );
        await demo.train(trainIterations, batchSize, numTestExamples);            // 开始训练
    });
}

runAdditionRNNDemo();
