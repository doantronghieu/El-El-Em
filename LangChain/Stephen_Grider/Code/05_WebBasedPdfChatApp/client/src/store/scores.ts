// import { writable } from './writeable';

export interface Scores {
	llm: {
		[name: string]: number[];
	};
	retriever: {
		[name: string]: number[];
	};
	memory: {
		[name: string]: number[];
	};
}

// const defaultState = {
// 	llm: {},
// 	retriever: {},
// 	memory: {}
// };

// const scores = writable<Scores>(defaultState);

// export const fetchScores = async () => {
// 	const response = await fetch('/chat/scores');
// 	const data = await response.json();
// 	scores.set(data);
// };

// export default scores;
