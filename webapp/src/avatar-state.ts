// State shared between the avatar picker, the streaming voice and the two
// live face features. In the single-file demo these were top-level `let`s
// that every block read and wrote; keeping them in one import-free module
// gives the same sharing without a dependency cycle.
export interface Avatar {
  id: string;
  name: string;
  category: string;
  image_url: string;
}

// Either a curated avatar id or an uploaded face as raw base64.
export type LiveSource = { avatar_id: string } | string | null;

export const avatarState = {
  all: [] as Avatar[],
  selected: null as string | null,
  liveSource: null as LiveSource,
};
