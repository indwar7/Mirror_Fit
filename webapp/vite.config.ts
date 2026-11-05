import { defineConfig } from 'vite';
import { execSync } from 'node:child_process';

// The build stamp used to be a string edited by hand ('build 73860f7+1').
// Reading it from git means the console line always names the real commit.
function gitStamp(): string {
  try { return execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(); }
  catch { return 'dev'; }
}

export default defineConfig({
  define: { __BUILD__: JSON.stringify('build ' + gitStamp()) },
  server: { port: 5173 },
  build: {
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      output: {
        // three.js and the MediaPipe runtime change on their own schedule;
        // keeping them in their own chunks means an app change does not
        // re-download them.
        manualChunks: { three: ['three'], mediapipe: ['@mediapipe/tasks-vision'] },
      },
    },
  },
});
