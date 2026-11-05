// Avatar picker for Live Face Swap.
import { FACE_API } from './config';
import { avatarState, type Avatar } from './avatar-state';
import { byId } from './dom';
import { setStatus } from './ui';
import { resetVoiceUI } from './voice';
import { checkLiveReady } from './liveswap';

export async function loadAvatars(): Promise<void> {
  try {
    const res  = await fetch(`${FACE_API}/avatars`);
    const data = (await res.json()) as { avatars: Avatar[] };
    avatarState.all = data.avatars;
    renderAvatars(avatarState.all);
  } catch (e) { console.warn('Avatar load failed:', e); }
}

export function renderAvatars(list: Avatar[]): void {
  // Cache-bust avatar thumbnails so stale randomuser.me images don't show.
  const v = Date.now();
  byId('avatarGrid').innerHTML = list.map((a) => `
    <div class="av-card${avatarState.selected === a.id ? ' selected' : ''}" id="av_${a.id}" onclick="selectAvatar('${a.id}')">
      <img src="${FACE_API}${a.image_url}?v=${v}" loading="lazy" onerror="this.src=''">
      <div class="av-name">${a.name}</div>
      <div class="av-check"><span>✓</span></div>
    </div>`).join('');
}

export function filterAvatars(cat: string, btn: HTMLElement): void {
  document.querySelectorAll('.pill').forEach((b) => b.classList.remove('active'));
  btn.classList.add('active');
  renderAvatars(cat === 'all' ? avatarState.all : avatarState.all.filter((a) => a.category === cat));
}

export function selectAvatar(id: string): void {
  avatarState.selected = id;
  const activeBtn = document.querySelector('.pill.active');
  const activeCat = activeBtn ? activeBtn.textContent : 'All';
  renderAvatars(activeCat === 'All' ? avatarState.all : avatarState.all.filter((a) => a.category === activeCat));

  const av = avatarState.all.find((a) => a.id === id);
  if (!av) return;

  const lsp = byId<HTMLImageElement>('lsp');
  lsp.src = `${FACE_API}/avatars/${id}/image`;
  lsp.style.display = 'block';
  byId('lsph').style.display = 'none';
  avatarState.liveSource = { avatar_id: id };
  checkLiveReady();
  setStatus('liveStatus', 'success', `${av.name} selected`);

  byId('voiceCard').style.display = 'block';
  byId('voiceAvatarName').textContent = av.name;
  resetVoiceUI();
}

export function setLiveSourceFile(inp: HTMLInputElement): void {
  avatarState.selected = null;
  document.querySelectorAll('.av-card').forEach((c) => c.classList.remove('selected'));
  byId('voiceCard').style.display = 'none';
  resetVoiceUI();
  const f = inp.files && inp.files[0]; if (!f) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    avatarState.liveSource = (e.target!.result as string).split(',')[1];
    checkLiveReady();
  };
  reader.readAsDataURL(f);
}
