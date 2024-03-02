<script lang="ts">
	import { onMount } from 'svelte';
	import Chart from 'chart.js/auto';

	export let data: { [key: string]: number[] };
	export let startingColor: { r: number; g: number; b: number };

	let chartCanvas: HTMLCanvasElement;

	onMount(() => {
		const ctx = chartCanvas.getContext('2d');
		if (!ctx) {
			return;
		}

		const labels = Object.keys(data);
		const chartValues = Object.values(data).map(
			(scores) => scores.reduce((a, b) => a + b, 0) / scores.length
		);

		new Chart(ctx, {
			type: 'bar',
			options: {
				plugins: {
					legend: { display: false }
				},
				scales: {
					y: {
						min: -1,
						max: 1,
						grid: {
							lineWidth: ({ tick }) => (tick.value == 0 ? 2 : 1),
							color: ({ tick }) => (tick.value === 0 ? 'rgba(0, 0, 0, 0.7)' : 'rgba(0, 0, 0, 0.1)')
						},
						ticks: {
							stepSize: 0.33,
							font: {
								size: 15
							}
						}
					},
					x: {
						ticks: {
							font: {
								size: 20
							}
						}
					}
				}
			},
			data: {
				labels: labels,
				datasets: [
					{
						base: 0,
						label: '',
						data: chartValues,
						backgroundColor: generateColors(startingColor, 7, 0.2),
						borderColor: generateColors(startingColor, 7),
						borderWidth: 1
					}
				]
			}
		});
	});

	function rgbToHsl(r: number, g: number, b: number) {
		r /= 255;
		g /= 255;
		b /= 255;
		let max = Math.max(r, g, b),
			min = Math.min(r, g, b);
		let h: number = (max + min) / 2;
		let s: number = (max + min) / 2;
		let l: number = (max + min) / 2;

		if (max === min) {
			h = s = 0; // achromatic
		} else {
			let diff = max - min;
			s = l > 0.5 ? diff / (2 - max - min) : diff / (max + min);
			switch (max) {
				case r:
					h = (g - b) / diff + (g < b ? 6 : 0);
					break;
				case g:
					h = (b - r) / diff + 2;
					break;
				case b:
					h = (r - g) / diff + 4;
					break;
			}
			h /= 6;
		}

		return [h * 360, s * 100, l * 100];
	}

	function hslToRgb(h: number, s: number, l: number) {
		let r, g, b;

		if (s === 0) {
			r = g = b = l; // achromatic
		} else {
			let hue2rgb = (p: number, q: number, t: number) => {
				if (t < 0) t += 1;
				if (t > 1) t -= 1;
				if (t < 1 / 6) return p + (q - p) * 6 * t;
				if (t < 1 / 2) return q;
				if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
				return p;
			};
			let q = l < 0.5 ? l * (1 + s) : l + s - l * s;
			let p = 2 * l - q;
			r = hue2rgb(p, q, h + 1 / 3);
			g = hue2rgb(p, q, h);
			b = hue2rgb(p, q, h - 1 / 3);
		}

		return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
	}

	function generateColors(rgb: { r: number; b: number; g: number }, k: number, alpha = 1) {
		const result = [];
		const [h, s, l] = rgbToHsl(rgb.r, rgb.g, rgb.b);
		const increment = 360 / k;

		for (let i = 0; i < k; i++) {
			let newHue = (h + increment * i) % 360;
			let [r, g, b] = hslToRgb(newHue / 360, s / 100, l / 100);
			result.push(`rgb(${r}, ${g}, ${b}, ${alpha})`);
		}

		return result;
	}

	const colors = generateColors({ r: 255, g: 99, b: 132 }, 7);
</script>

<canvas bind:this={chartCanvas} />
