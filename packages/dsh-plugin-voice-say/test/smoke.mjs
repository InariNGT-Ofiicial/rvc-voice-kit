// 离线冒烟：用假 ctx 调 voice_say，验证「两个常驻服务 + 落盘到 agent cwd」这条链路真的能跑。
// 用法：node test/smoke.mjs [cwd]
import { existsSync, statSync } from 'node:fs';
import { apply } from '../src/index.js';

const cwd = process.argv[2] ?? 'D:/WorkData/qqbot-workspace/tts-test-rvc-research';
const registry = new Map();
const ctx = {
  get: (svc) => (svc === 'tools' ? { register: (def) => registry.set(def.name, def) } : undefined),
  logger: console,
};

await apply(ctx, {});
console.log('registered:', [...registry.keys()].join(', '));

const doctor = await registry.get('voice_doctor').execute({}, { cwd });
console.log('doctor:', JSON.stringify(doctor));

const t0 = Date.now();
const r = await registry.get('voice_say').execute({ text: '今天天气真不错，我们出去走走吧。' }, { cwd });
const wall = Date.now() - t0;
console.log('result:', JSON.stringify(r));
console.log('wall_ms:', wall, 'exists:', existsSync(r.file_path), 'bytes:', statSync(r.file_path).size);

const rendered = registry.get('voice_say').output.render({}, r);
console.log('render:', rendered[0].text);

// 负数用例：超长文本必须被拒
try {
  await registry.get('voice_say').execute({ text: '啊'.repeat(200) }, { cwd });
  console.log('FAIL: 超长文本竟然通过了');
  process.exitCode = 1;
} catch (e) {
  console.log('长文本拒绝 OK:', e.message);
}
