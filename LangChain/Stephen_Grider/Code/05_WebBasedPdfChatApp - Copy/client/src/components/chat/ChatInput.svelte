<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	let value = '';

	const dispatch = createEventDispatcher();
	function handleKeyDown(event: KeyboardEvent) {
		const isCombo = event.shiftKey || event.ctrlKey || event.altKey || event.metaKey;
		if (event.key !== 'Enter' || isCombo) {
			return;
		}

		if (event.key === 'Enter' && !isCombo && value === '') {
			event.preventDefault();
			return;
		}

		event.preventDefault();
		dispatch('submit', value);
		value = '';
	}

	$: height = (value.match(/\n/g)?.length || 0) * 25 + 72;
</script>

<textarea
	class="w-full mx-auto py-1.5 px-2.5 resize-none border rounded max-h-40"
	style:height={height + 'px'}
	bind:value
	on:keydown={handleKeyDown}
/>
