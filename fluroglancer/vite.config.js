import packageJson from "./package.json"

export default {
    root: 'src/',
    publicDir: '../static/',
    base: './',
    server:
    {
        host: false,
        open: false,
        fs: {
            deny: ['.env', '.env.*', '*.{crt,pem}', '**/.git/**']
        }
    },
    build:
    {
        outDir: '../build/v' + packageJson.version + '/',
        emptyOutDir: true,
        sourcemap: true
    },
}
