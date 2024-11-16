module.exports = {
    apps: [
        {
            name: 'weldscanner',
            script: 'app.js',
            node_args: '--max-old-space-size=4096',
        },
        {
            name: 'weldscanner-trainer',
            script: 'models/trainer.js',
            node_args: '--max-old-space-size=4096',
        }
    ]
}